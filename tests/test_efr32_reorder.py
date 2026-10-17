"""Hidden-neuron reordering (spikeduet/backend/efr32/reorder.py) and the hybrid build it targets -- range-sparse MVP
hidden layer with flash-resident FP16 weights, event-driven M33 input/output layers, folded output affine."""
import numpy as np
import pytest
import torch

from spikeduet.backend.efr32.codegen import Efr32Backend
from spikeduet.backend.efr32.reorder import cofiring_order, permute_hidden_neurons
from spikeduet.ir.verifier import verify_module_ir
from spikeduet.reference.simulator import run_sequence
from tests.test_efr32_cpu_sparse import _fold_module, _module

DIMS = [(24, 64), (64, 64), (64, 10)]


def test_cofiring_order_groups_coactive_neurons():
    rng = np.random.default_rng(0)
    n, groups = 48, 6
    member = rng.permutation(n) % groups                       # 6 hidden groups of 8 neurons, scattered over the indices
    s = np.zeros((600, n), bool)
    for t in range(600):
        s[t] = (member == rng.integers(groups)) & (rng.random(n) < 0.9)
    order = cofiring_order(s)
    assert sorted(order) == list(range(n))
    runs = 1 + int(np.sum(np.diff(member[order]) != 0))        # contiguous same-group runs in the new order
    assert runs == groups
    assert cofiring_order(s) == order                          # deterministic


def test_permutation_relabels_without_changing_the_network():
    ir, _ = _module(DIMS, steps=10, thr=0.3)
    order = np.random.default_rng(1).permutation(64).tolist()
    for hidden in (0, 1):
        ir2 = permute_hidden_neurons(ir, hidden, order)
        verify_module_ir(ir2)
        x = [torch.rand(5, 24) for _ in range(10)]
        _, _, t1 = run_sequence(ir, x)
        _, _, t2 = run_sequence(ir2, x)
        for a, b in zip(t1, t2):
            assert torch.equal(a.spikes[hidden][:, order], b.spikes[hidden])       # same neurons, new labels
            assert torch.allclose(a.mem[-1], b.mem[-1], atol=1e-5)                 # readout: reordered FP32 sums only


def test_permutation_rejects_bad_requests():
    ir, _ = _module(DIMS, steps=4)
    with pytest.raises(ValueError):
        permute_hidden_neurons(ir, 2, list(range(10)))             # the output layer
    with pytest.raises(ValueError):
        permute_hidden_neurons(ir, 0, [0] * 64)                    # not a permutation


def test_hybrid_range_sparse_flash_fold_structure():
    _, mod = _fold_module(DIMS)
    opts = dict(engine="mvp", sparse_mvp=True, force_sparse_mvp_layers=frozenset({1}), cpu_sparse_layers=frozenset({0, 2}),
                cpu_sparse_weight_placement={0: "sram", 2: "sram"}, mvp_flash_weight_layers=frozenset({1}), fold_output_affine=True)
    c = Efr32Backend().generate(mod, **opts).files["snn_generated_mvp.c"]
    assert "if (l == 0 || l == 2) {" in c and "} else if (iscas_w16_flash[l] != 0) {" in c
    assert "static const uint16_t iscas_w16_l1[" in c and "iscas_w16_l0[" not in c
    assert "affine layer 1 (MVP, range-merged sparse" in c and "FP16 source in place" in c
    assert "-> direct FP16 g_in_f16[1]" in c
    assert "output affine FOLDED" in c and "iscas_csp_acc2(" in c
    assert c.count("sl_math_mvp_matrix_vector_mult_f16(&g_mat[") == 1   # only the ISCAS_SPARSE_VERIFY recompute of layer 1
    with pytest.raises(ValueError):                                     # folded output affine must be an M33 layer
        Efr32Backend().generate(mod, engine="mvp", fold_output_affine=True)
    with pytest.raises(ValueError):                                     # a flash FP16 table for an M33 layer makes no sense
        Efr32Backend().generate(mod, **{**opts, "mvp_flash_weight_layers": frozenset({0, 1})})


# ---------------------------------------------------------------- the rule inside the automatic planner
from spikeduet.backend.efr32.placement import (ActivityProfile, LayerActivity, MVP_RANGE_REORDER, M33, ReorderCandidate,
                                              hidden_range_stats, plan_engine_placement, plan_with_reordering, reorder_candidate)


def _profile(dims, hidden_density, launches, pairs, order=None):
    layers = {0: LayerActivity(active_inputs=0.25 * dims[0][0], nonempty_fraction=0.9, binary=True)}
    for i, (n, _m) in enumerate(dims[1:], 1):
        layers[i] = LayerActivity(active_inputs=hidden_density * n, nonempty_fraction=0.95, binary=True)
    n1 = dims[1][0]
    cand = ReorderCandidate(order=tuple(order or reversed(range(n1))), launches=launches, pairs=pairs,
                            launches_as_trained=2 * launches, pairs_as_trained=1.3 * pairs, source="test")
    return ActivityProfile(layers=layers, source="test", num_samples=1, num_timesteps=12, reorder={1: cand})


def test_planner_selects_reordering_only_where_it_is_cheapest():
    wide = [(24, 384), (384, 384), (384, 10)]
    ir, mod = _fold_module(wide)
    prof = _profile(wide, 0.05, launches=5.8, pairs=27.0)
    base = plan_engine_placement(mod, prof)                              # rule off (default): the profile's candidate is ignored
    assert not base.reorder and MVP_RANGE_REORDER not in {d.engine for d in base.layers}
    assert base.to_dict() == plan_engine_placement(mod, ActivityProfile(prof.layers, "test", 1, 12)).to_dict()
    ir2, mod2, plan = plan_with_reordering(ir, prof, num_timesteps=12, fold_output=True)
    d1 = next(d for d in plan.layers if d.layer == 1)
    assert d1.engine == MVP_RANGE_REORDER and d1.candidates[MVP_RANGE_REORDER] < d1.candidates[M33]
    assert plan.mvp_flash_weight_layers == frozenset({1})                # 295 KB of FP16 weights cannot sit in SRAM
    assert plan.fold_output == 2 and next(d for d in plan.layers if d.layer == 2).forced == "fold"
    kw = plan.generate_kwargs()
    assert kw["force_sparse_mvp_layers"] == frozenset({1}) and kw["cpu_sparse_layers"] == frozenset({0, 2}) and kw["fold_output_affine"]
    order = list(plan.reorder[1])
    assert torch.equal(ir2.layers[0].weight, ir.layers[0].weight[order]) and torch.equal(ir2.layers[2].weight, ir.layers[2].weight[:, order])
    c = Efr32Backend().generate(mod2, **kw).files["snn_generated_mvp.c"]
    assert "affine layer 1 (MVP, range-merged sparse" in c and "iscas_w16_l1[" in c and "output affine FOLDED" in c

    small = [(24, 64), (64, 64), (64, 10)]                               # launch-bound layer: the same rule declines
    ir, mod = _fold_module(small)
    ir3, _mod3, plan = plan_with_reordering(ir, _profile(small, 0.12, launches=1.2, pairs=10.0), num_timesteps=12)
    assert not plan.reorder and ir3 is ir
    assert next(d for d in plan.layers if d.layer == 1).engine == M33


def test_reorder_candidate_is_scored_on_held_out_steps():
    rng = np.random.default_rng(0)
    member = rng.permutation(64) % 4
    def trace(k):
        return np.array([(member == rng.integers(4)) & (rng.random(64) < 0.8) for _ in range(k)])
    c = reorder_candidate(trace(400), trace(300), rows=64, source="test")
    assert sorted(c.order) == list(range(64))
    assert c.launches < c.launches_as_trained and c.pairs <= c.pairs_as_trained
    assert ActivityProfile.from_dict(ActivityProfile({}, "s", 1, 1, reorder={1: c}).to_dict()).reorder[1] == c
    assert hidden_range_stats(np.zeros((5, 64), bool), 64) == (0.0, 0.0)
