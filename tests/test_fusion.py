"""IR-level tests for `spikeduet.ir.fusion.SpikeStagingFusionPass`
-- legality (positive/negative), statistics, dead-buffer removal, and
composition with LICM, all at the Execution IR level (no C compilation;
see `tests/test_backend_licm.py`-style files for host-compiled conformance).
"""

import dataclasses


from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.ir.execution import AffineOp, BufferKind, ConvertOp, LIFUpdateOp, TemporalMode
from spikeduet.ir.execution_verifier import verify_execution_ir
from spikeduet.ir.fusion import SpikeStagingFusionPass, find_fusion_candidate
from spikeduet.ir.lowering import lower_source_ir
from spikeduet.ir.pass_manager import (
    baseline_pipeline,
    compile_source_to_execution_ir,
    fusion_pipeline,
    licm_pipeline,
    optimized_pipeline,
)
from tests.test_reference import ChainSNN


def _module(dims, *, num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT, **kw):
    n = len(dims)
    betas = kw.pop("betas", [0.9] * n)
    thresholds = kw.pop("thresholds", [1.0] * n)
    model = ChainSNN(dims=list(dims), betas=betas, thresholds=thresholds, **kw)
    ir = parse_snntorch_model(model)
    return lower_source_ir(ir, num_timesteps=num_timesteps, input_temporal_mode=input_temporal_mode)


# --- positive: single consumer fuses ----------------------------------------


def test_single_intermediate_boundary_fuses():
    module = _module([(4, 6), (6, 3)])
    pass_ = SpikeStagingFusionPass()
    fused = pass_.run(module)

    assert pass_.stats.candidates_considered == 2  # lif0, lif1
    assert pass_.stats.fusions_applied == 1  # only lif0 -> convert1 -> affine1 is fusable
    assert pass_.stats.convert_ops_removed == 1
    assert pass_.stats.buffers_eliminated == 1  # spike0 dropped

    ops_by_id = {op.id: op for op in fused.loop[0].body}
    assert "convert1" not in ops_by_id
    lif0 = ops_by_id["lif0"]
    assert isinstance(lif0, LIFUpdateOp)
    assert lif0.output_spike == "stage1"

    affine1 = ops_by_id["affine1"]
    assert isinstance(affine1, AffineOp)
    assert affine1.input == "stage1"  # unchanged -- still reads the same buffer

    buffer_names = {b.name for b in fused.buffers}
    assert "spike0" not in buffer_names
    assert "stage1" in buffer_names  # still exists, now LIF-written


def test_fused_module_passes_verifier():
    module = _module([(4, 6), (6, 3)])
    fused = SpikeStagingFusionPass().run(module)
    verify_execution_ir(fused)  # must not raise


def test_no_fusion_leaves_module_unchanged_object():
    """A module with nothing fusable (e.g. single-layer, only the final
    output boundary) is returned as-is."""
    module = _module([(4, 3)])
    pass_ = SpikeStagingFusionPass()
    result = pass_.run(module)
    assert result is module
    assert pass_.stats.fusions_applied == 0


# --- negative: final output layer is never fused -----------------------------


def test_final_output_layer_never_fused():
    module = _module([(4, 6), (6, 3)])
    fused = SpikeStagingFusionPass().run(module)

    # the last layer's LIF still writes its own spike buffer, still the
    # module's output_buffer, still readable/correct as the readout source
    assert fused.output_buffer == "spike1"
    ops_by_id = {op.id: op for op in fused.loop[0].body}
    lif1 = ops_by_id["lif1"]
    assert lif1.output_spike == "spike1"
    assert fused.get_buffer("spike1").kind is BufferKind.SPIKE


def test_candidate_search_rejects_output_buffer():
    module = _module([(4, 3)])
    lif0 = next(op for op in module.loop[0].body if isinstance(op, LIFUpdateOp))
    assert find_fusion_candidate(module, lif0) is None


# --- negative: multiple consumers block fusion --------------------------------


def test_multiple_consumers_block_fusion():
    """Hand-construct a second (debug) ConvertOp reading spike0 -- now
    spike0 has TWO consumers, so fusion must not apply."""
    module = _module([(4, 6), (6, 3)])
    loop = module.loop[0]
    extra = ConvertOp(id="debug_convert0", input="spike0", output="stage1")
    # give the extra a distinct, otherwise-unused output name to avoid an
    # unrelated verifier complaint about two producers of the same buffer
    extra = dataclasses.replace(extra, output="debug_stage0")
    new_buffers = module.buffers + (
        dataclasses.replace(module.get_buffer("stage1"), name="debug_stage0"),
    )
    new_body = loop.body + (extra,)
    module2 = dataclasses.replace(
        module, buffers=new_buffers, loop=(dataclasses.replace(loop, body=new_body),)
    )

    pass_ = SpikeStagingFusionPass()
    result = pass_.run(module2)

    assert pass_.stats.fusions_applied == 0
    ops_by_id = {op.id: op for op in result.loop[0].body}
    assert "convert1" in ops_by_id  # original staging convert still present
    assert "debug_convert0" in ops_by_id
    lif0 = ops_by_id["lif0"]
    assert lif0.output_spike == "spike0"  # unchanged -- unsafe to fuse


# --- pipelines: composition with LICM -----------------------------------------


def test_licm_then_fusion_compose_on_mnist_like():
    ir = parse_snntorch_model(
        ChainSNN(dims=[(8, 6), (6, 5), (5, 3)], betas=[0.9, 0.9, 0.9], thresholds=[1.0, 1.0, 1.0])
    )
    module = compile_source_to_execution_ir(
        ir, num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT, pipeline=optimized_pipeline()
    )

    # LICM: layer0's convert+affine hoisted
    preheader_ids = {op.id for op in module.preheader_ops}
    assert preheader_ids == {"convert0", "affine0"}

    # fusion: both intermediate boundaries (lif0->affine1, lif1->affine2) fused
    loop_ids = {op.id for op in module.loop[0].body}
    assert "convert1" not in loop_ids
    assert "convert2" not in loop_ids

    lif_by_id = {op.id: op for op in module.loop[0].body if isinstance(op, LIFUpdateOp)}
    assert lif_by_id["lif0"].output_spike == "stage1"
    assert lif_by_id["lif1"].output_spike == "stage2"
    assert lif_by_id["lif2"].output_spike == "spike2"  # final output, never fused

    verify_execution_ir(module)


def test_licm_finds_nothing_to_hoist_on_shd_like_fusion_still_applies():
    ir = parse_snntorch_model(
        ChainSNN(dims=[(8, 6), (6, 3)], betas=[0.9, 0.9], thresholds=[1.0, 1.0])
    )
    module = compile_source_to_execution_ir(
        ir, num_timesteps=5, input_temporal_mode=TemporalMode.VARYING, pipeline=optimized_pipeline()
    )

    assert module.preheader_ops == ()  # nothing hoistable under temporal input

    loop_ids = {op.id for op in module.loop[0].body}
    assert "convert0" in loop_ids  # layer 0's own staging (from raw input) is never fused
    assert "convert1" not in loop_ids  # lif0 -> affine1 boundary IS fused

    lif0 = next(op for op in module.loop[0].body if op.id == "lif0")
    assert lif0.output_spike == "stage1"

    verify_execution_ir(module)


def test_all_four_pipelines_available_and_distinct():
    ir = parse_snntorch_model(
        ChainSNN(dims=[(4, 6), (6, 3)], betas=[0.9, 0.9], thresholds=[1.0, 1.0])
    )
    kwargs = dict(num_timesteps=5, input_temporal_mode=TemporalMode.INVARIANT)

    base = compile_source_to_execution_ir(ir, pipeline=baseline_pipeline(), **kwargs)
    licm = compile_source_to_execution_ir(ir, pipeline=licm_pipeline(), **kwargs)
    fusion = compile_source_to_execution_ir(ir, pipeline=fusion_pipeline(), **kwargs)
    both = compile_source_to_execution_ir(ir, pipeline=optimized_pipeline(), **kwargs)

    assert base.preheader_ops == () and _has_convert1(base)
    assert licm.preheader_ops != () and _has_convert1(licm)
    assert fusion.preheader_ops == () and not _has_convert1(fusion)
    assert both.preheader_ops != () and not _has_convert1(both)


def _has_convert1(module) -> bool:
    return any(op.id == "convert1" for op in module.walk_ops())


# --- arbitrary depth ----------------------------------------------------------


def test_arbitrary_depth_all_intermediate_boundaries_fuse():
    module = _module([(6, 8), (8, 6), (6, 5), (5, 3)], num_timesteps=6)
    fused = SpikeStagingFusionPass().run(module)

    loop_ids = {op.id for op in fused.loop[0].body}
    # 4 layers -> 3 intermediate boundaries fusable (convert1, convert2, convert3);
    # convert0 (raw input staging) is never a fusion target (no upstream LIF)
    for i in (1, 2, 3):
        assert f"convert{i}" not in loop_ids
    assert "convert0" in loop_ids

    lif_by_id = {op.id: op for op in fused.loop[0].body if isinstance(op, LIFUpdateOp)}
    assert lif_by_id["lif0"].output_spike == "stage1"
    assert lif_by_id["lif1"].output_spike == "stage2"
    assert lif_by_id["lif2"].output_spike == "stage3"
    assert lif_by_id["lif3"].output_spike == "spike3"  # final output
    verify_execution_ir(fused)


# --- all Leaky semantics untouched by the rewrite (structural, not numeric --
# see tests/test_backend_licm.py for host-compiled numeric conformance) ------


def test_fusion_preserves_lif_semantics_fields():
    from snntorch.functional import quant

    q = quant.state_quant(num_bits=3, uniform=True, threshold=1.0)
    module = _module(
        [(4, 4), (4, 3)],
        betas=[[0.0, 0.4, 0.8, 1.0], 0.9],
        thresholds=[[0.3, 0.6, 0.9, 1.2], 1.0],
        graded_spikes_factors=[[0.25, 0.75, 1.5, 2.5], 1.0],
        reset_mechanisms=["zero", "subtract"],
        reset_delays=[True, False],
        inhibitions=[True, False],
        state_quants=[q, False],
    )
    before_semantics = next(
        op.semantics for op in module.loop[0].body if isinstance(op, LIFUpdateOp) and op.id == "lif0"
    )
    fused = SpikeStagingFusionPass().run(module)
    after_semantics = next(
        op.semantics for op in fused.loop[0].body if isinstance(op, LIFUpdateOp) and op.id == "lif0"
    )
    # fusion only ever rebuilds output_spike -- semantics object identical
    assert after_semantics == before_semantics
