"""NIR frontend (compiler/frontend/nir_parser.py): R1 operator coverage + rejections, R2 round-trip exactness."""
import numpy as np
import pytest
import torch

nir = pytest.importorskip("nir")

from spikeduet.backend.api import compile_model
from spikeduet.frontend.nir_parser import NIRImportError, parse_nir_graph, snntorch_chain_to_nir
from spikeduet.frontend.snntorch_parser import parse_snntorch_model
from spikeduet.ir.execution import TemporalMode
from spikeduet.ir.model import LinearLayer, LIFLayer
from spikeduet.reference.simulator import readout_scores, run_sequence
from tests.backend_c_harness import CC, compile_portable_c, python_reference_predict, run_harness
from tests.test_reference import ChainSNN

DT = 1e-4


def _scores(ir, xs):
    _, _, traces = run_sequence(ir, xs)
    return readout_scores(ir, traces)


def _chain_model(resets, betas, thresholds, seed=0, dims=((6, 5), (5, 3))):
    torch.manual_seed(seed)
    return ChainSNN(dims=list(dims), betas=betas, thresholds=thresholds, reset_mechanisms=resets).eval()


def _graph(nodes, names):
    return nir.NIRGraph(nodes=nodes, edges=list(zip(names[:-1], names[1:])))


def _lif(n, tau=2e-3, thr=1.0, v_reset=0.0, r=None):
    """`n` is a neuron count or a shape such as (C, H, W) after a conv."""
    tau = np.full(n, tau, np.float32)
    return nir.LIF(tau=tau, r=tau / DT if r is None else np.full(n, r, np.float32), v_leak=np.zeros(n, np.float32),
                   v_threshold=np.full(n, thr, np.float32), v_reset=np.full(n, v_reset, np.float32))


def _affine(n_out, n_in, seed=0):
    g = np.random.default_rng(seed)
    return nir.Affine(weight=g.normal(size=(n_out, n_in)).astype(np.float32), bias=g.normal(size=n_out).astype(np.float32))


# ------------------------------------------------------------------ R2: round trips
def test_zero_reset_model_round_trips_through_stock_nir_semantics():
    net = _chain_model(["zero", "zero"], [torch.tensor([0.3, 0.5, 0.7, 0.9, 0.95]), 0.8], [1.0, torch.tensor([0.5, 1.0, 1.5])])
    direct = parse_snntorch_model(net)
    via = parse_nir_graph(snntorch_chain_to_nir(net), dt=DT, semantics="nir")
    for a, b in zip(direct.layers, via.layers):
        if isinstance(a, LinearLayer):
            assert torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias)
        else:
            assert (a.reset_mechanism, a.reset_delay) == (b.reset_mechanism, b.reset_delay)
            assert np.allclose(np.broadcast_to(a.beta, len(b.beta)), b.beta, atol=1e-6)
            assert np.broadcast_to(a.threshold, len(b.threshold)).tolist() == list(b.threshold) or a.threshold == b.threshold
    torch.manual_seed(1)
    xs = [torch.randn(16, 6) * 2 for _ in range(12)]
    s1, s2 = _scores(direct, xs), _scores(via, xs)
    assert torch.equal(s1.argmax(1), s2.argmax(1))
    assert float((s1 - s2).abs().max()) < 1e-4


def test_annotated_round_trip_is_lossless_through_a_file(tmp_path):
    net = _chain_model(["subtract", "none"], [0.9, torch.tensor([0.2, 0.6, 1.0])], [1.0, 0.7])
    p = tmp_path / "g.nir"
    with np.errstate(divide="ignore", invalid="ignore"):
        nir.write(str(p), snntorch_chain_to_nir(net, annotate=True))
    direct = parse_snntorch_model(net)
    rep = {}
    via = parse_nir_graph(nir.read(str(p)), dt=DT, semantics="annotated", report=rep)
    assert len(rep["annotated"]) == 2
    for a, b in zip(direct.layers, via.layers):
        if isinstance(a, LIFLayer):
            assert (a.beta, a.threshold, a.reset_mechanism, a.reset_delay) == (b.beta, b.threshold, b.reset_mechanism, b.reset_delay)
    torch.manual_seed(2)
    xs = [torch.randn(8, 6) * 2 for _ in range(10)]
    assert torch.equal(_scores(direct, xs), _scores(via, xs))


def test_spec_semantics_turns_subtract_into_zero_reset():
    net = _chain_model(["subtract", "subtract"], [0.9, 0.9], [1.0, 1.0])
    via = parse_nir_graph(snntorch_chain_to_nir(net), dt=DT)
    assert [l.reset_mechanism for l in via.layers if isinstance(l, LIFLayer)] == ["zero", "zero"]


def test_beta_one_is_refused_by_default_and_clamped_on_request():
    net = _chain_model(["zero", "zero"], [torch.tensor([1.0, 0.5, 1.0, 0.5, 0.5]), 0.9], [1.0, 1.0])
    g = snntorch_chain_to_nir(net)
    with pytest.raises(NIRImportError, match="time constant"):
        parse_nir_graph(g, dt=DT)
    rep = {}
    ir = parse_nir_graph(g, dt=DT, nonphysical="clamp", report=rep)
    assert rep["clamped"] == [{"node": "lifs.0", "neurons": 2}]
    assert ir.layers[1].beta[0] == 1.0 and ir.layers[1].beta[2] == 1.0
    assert torch.equal(ir.layers[0].weight, parse_snntorch_model(net).layers[0].weight)    # s -> 1 in the limit


@pytest.mark.skipif(CC is None, reason="no host C compiler")
def test_generated_portable_c_from_a_nir_graph_matches_the_reference(tmp_path):
    net = _chain_model(["zero", "zero"], [0.85, torch.tensor([0.5, 0.9, 0.99])], [1.0, 1.0])
    ir = parse_nir_graph(snntorch_chain_to_nir(net), dt=DT)
    art = compile_model(ir, backend="portable_c", num_timesteps=6, input_temporal_mode=TemporalMode.INVARIANT, include_harness=True)
    exe = compile_portable_c(art.files, tmp_path)
    torch.manual_seed(3)
    samples = [torch.randn(1, 6) * 2 for _ in range(8)]
    assert run_harness(exe, [x[0].tolist() for x in samples]) == [python_reference_predict(ir, [x] * 6) for x in samples]


# ------------------------------------------------------------------ R1: operators
def test_conv_pool_flatten_lowering_matches_torch():
    g0 = np.random.default_rng(4)
    w = g0.normal(size=(3, 2, 3, 3)).astype(np.float32)
    b = g0.normal(size=3).astype(np.float32)
    nodes = {"input": nir.Input(input_type=np.array([2, 6, 6])),
             "conv": nir.Conv2d(input_shape=(6, 6), weight=w, stride=1, padding=1, dilation=1, groups=1, bias=b),
             "lif0": _lif((3, 6, 6)),
             "pool": nir.AvgPool2d(kernel_size=np.array([2, 2]), stride=np.array([2, 2]), padding=np.array([0, 0])),
             "flat": nir.Flatten(input_type={"input": np.array([3, 3, 3])}, start_dim=0, end_dim=-1),
             "fc": _affine(4, 27, seed=5), "out": nir.LI(tau=np.full(4, 1e-3, np.float32), r=np.full(4, 10.0, np.float32),
                                                       v_leak=np.zeros(4, np.float32)),
             "output": nir.Output(output_type=np.array([4]))}
    ir = parse_nir_graph(_graph(nodes, list(nodes)), dt=DT, nonphysical="error")
    assert ir.layers[-1].reset_mechanism == "none"
    x = torch.from_numpy(g0.normal(size=(5, 2, 6, 6)).astype(np.float32))
    ref = torch.nn.functional.conv2d(x, torch.from_numpy(w), torch.from_numpy(b), padding=1).flatten(1)
    s = 2e-3 / DT * DT / 2e-3                                       # r dt / tau of _lif = 1
    assert torch.allclose(x.flatten(1) @ ir.layers[0].weight.T + ir.layers[0].bias, ref * s, atol=1e-5)
    # the second Linear is pool -> flatten -> fc, composed; check it on an arbitrary spike map
    spk = (torch.rand(5, 3, 6, 6) > 0.5).float()
    fc = nodes["fc"]
    ref2 = torch.nn.functional.avg_pool2d(spk, 2).flatten(1) @ torch.from_numpy(fc.weight).T + torch.from_numpy(fc.bias)
    li_s = 10.0 * DT / 1e-3
    assert torch.allclose(spk.flatten(1) @ ir.layers[2].weight.T + ir.layers[2].bias, ref2 * li_s, atol=1e-4)


def test_if_node_is_a_beta_one_integrator():
    nodes = {"input": nir.Input(input_type=np.array([3])), "fc": _affine(2, 3),
             "if": nir.IF(r=np.ones(2, np.float32), v_threshold=np.ones(2, np.float32), v_reset=np.zeros(2, np.float32)),
             "output": nir.Output(output_type=np.array([2]))}
    ir = parse_nir_graph(_graph(nodes, list(nodes)), dt=DT)
    assert ir.layers[1].beta == [1.0] and ir.layers[1].reset_mechanism == "zero"


@pytest.mark.parametrize("case,match", [
    ("cubalif", "unsupported NIR node type CubaLIF"),
    ("v_reset", "v_reset != 0"),
    ("li_middle", "LI .* only supported as the last"),
    ("trailing_linear", "ends with linear operators"),
    ("branch", "branching"),
    ("neuron_first", "not preceded by a linear operator"),
    ("big_conv", "Toeplitz lowering would need"),
])
def test_unsupported_graphs_fail_loudly(case, match):
    n = 3
    nodes = {"input": nir.Input(input_type=np.array([4])), "fc": _affine(n, 4), "lif": _lif(n),
             "output": nir.Output(output_type=np.array([n]))}
    names = list(nodes)
    if case == "cubalif":
        nodes["lif"] = nir.CubaLIF(tau_syn=np.full(n, 1e-3), tau_mem=np.full(n, 2e-3), r=np.full(n, 20.0), v_leak=np.zeros(n),
                                   v_threshold=np.ones(n), v_reset=np.zeros(n), w_in=np.ones(n))
    elif case == "v_reset":
        nodes["lif"] = _lif(n, v_reset=-0.5)
    elif case == "li_middle":
        nodes = {"input": nodes["input"], "fc": nodes["fc"],
                 "li": nir.LI(tau=np.full(n, 2e-3), r=np.full(n, 20.0), v_leak=np.zeros(n)),
                 "fc2": _affine(2, n), "lif": _lif(2), "output": nir.Output(output_type=np.array([2]))}
        names = list(nodes)
    elif case == "trailing_linear":
        nodes = {"input": nodes["input"], "fc": nodes["fc"], "lif": nodes["lif"], "fc2": _affine(2, n),
                 "output": nir.Output(output_type=np.array([2]))}
        names = list(nodes)
    elif case == "neuron_first":
        nodes = {"input": nir.Input(input_type=np.array([n])), "lif": _lif(n), "output": nir.Output(output_type=np.array([n]))}
        names = list(nodes)
    elif case == "big_conv":
        nodes = {"input": nir.Input(input_type=np.array([2, 128, 128])),
                 "conv": nir.Conv2d(input_shape=(128, 128), weight=np.ones((16, 2, 3, 3), np.float32), stride=1, padding=1,
                                    dilation=1, groups=1, bias=np.zeros(16, np.float32)),
                 "lif": _lif((16, 128, 128)), "output": nir.Output(output_type=np.array([16, 128, 128]))}
        names = list(nodes)
    if case == "branch":
        g = nir.NIRGraph(nodes=nodes, edges=[("input", "fc"), ("fc", "lif"), ("lif", "output"), ("fc", "output")])
    else:
        g = _graph(nodes, names)
    with pytest.raises(NIRImportError, match=match):
        parse_nir_graph(g, dt=DT)


# ------------------------------------------------------------------ R4: a model from another framework (Sinabs)
def _sinabs():
    return pytest.importorskip("sinabs")


def _dyadic(shape, gen, lo=-128, hi=128, den=64.0):
    """Weights on a 1/64 grid: every product and sum below stays exact in float32, so summation order
    cannot hide a semantic difference and the comparisons can be bitwise."""
    return torch.randint(lo, hi, shape, generator=gen).float() / den


def _sinabs_run(net, xs):
    """Sinabs reference: one timestep per call (state kept), summed output-layer membrane after reset."""
    import sinabs
    sinabs.reset_states(net)
    total = None
    for x in xs:
        net(x)
        v = net[-1].v_mem.reshape(x.shape[0], -1).clone()
        total = v if total is None else total + v
    return total


def _via_file(graph, tmp_path):
    """Sinabs 3.1.3 graphs are parsed in memory: with nir 1.0.8 a file round trip fails (the Input node keeps a
    batch dim of 1, which NIR's type inference on read rejects; conv graphs carry fields h5py cannot store)."""
    return graph


@pytest.mark.skipif(CC is None, reason="no host C compiler")
def test_sinabs_iaf_mlp_is_bit_exact_through_nir_and_portable_c(tmp_path):
    _sinabs()
    import sinabs.layers as sl
    from sinabs import nir as snir
    from sinabs.activation import MembraneReset, SingleSpike
    from tests.backend_c_harness import float_bits, run_memsum
    g = torch.Generator().manual_seed(7)
    thr = 1.0 + 1.0 / 1024                      # never hit exactly by sums on the 1/512 grid (sinabs fires on >=, NIR on >)
    iaf = lambda: sl.IAF(spike_threshold=torch.tensor(thr), spike_fn=SingleSpike, reset_fn=MembraneReset())
    net = torch.nn.Sequential(torch.nn.Linear(6, 8), iaf(), torch.nn.Linear(8, 3), iaf())
    with torch.no_grad():
        for lin in (net[0], net[2]):
            lin.weight.copy_(_dyadic(lin.weight.shape, g))
            lin.bias.copy_(_dyadic(lin.bias.shape, g, -32, 32))
    graph = _via_file(snir.to_nir(net, torch.zeros(1, 1, 6)), tmp_path)
    ir = parse_nir_graph(graph, dt=1.0)
    assert [l.beta for l in ir.layers if isinstance(l, LIFLayer)] == [[1.0], [1.0]]
    T, B = 10, 12
    xs = [torch.randint(0, 9, (B, 6), generator=g).float() / 8 for _ in range(T)]
    with torch.no_grad():
        ref = _sinabs_run(net, [x[:, None, :] for x in xs])
    ours = _scores(ir, xs)
    assert float_bits(ref) == float_bits(ours)
    art = compile_model(ir, backend="portable_c", num_timesteps=T, input_temporal_mode=TemporalMode.VARYING)
    flat = [torch.stack([x[b] for x in xs]).reshape(-1).tolist() for b in range(B)]
    assert run_memsum(art.files, tmp_path / "c", flat) == [float_bits(r) for r in ref]


@pytest.mark.skipif(CC is None, reason="no host C compiler")
def test_sinabs_conv_iaf_net_is_bit_exact_through_toeplitz_lowering(tmp_path):
    _sinabs()
    import sinabs.layers as sl
    from sinabs import nir as snir
    from sinabs.activation import MembraneReset, SingleSpike
    from tests.backend_c_harness import float_bits, run_memsum
    g = torch.Generator().manual_seed(11)
    thr = torch.tensor(1.0 + 1.0 / 1024)
    iaf = lambda: sl.IAFSqueeze(batch_size=1, spike_threshold=thr, spike_fn=SingleSpike, reset_fn=MembraneReset())
    net = torch.nn.Sequential(torch.nn.Conv2d(2, 3, 3, padding=1), iaf(), sl.SumPool2d(2), torch.nn.Flatten(),
                              torch.nn.Linear(27, 4), iaf())
    with torch.no_grad():
        net[0].weight.copy_(_dyadic(net[0].weight.shape, g)); net[0].bias.copy_(_dyadic((3,), g, -32, 32))
        net[4].weight.copy_(_dyadic(net[4].weight.shape, g)); net[4].bias.copy_(_dyadic((4,), g, -32, 32))
    graph = _via_file(snir.to_nir(net, torch.zeros(1, 2, 6, 6)), tmp_path)
    ir = parse_nir_graph(graph, dt=1.0)
    assert [l.in_features for l in ir.layers if isinstance(l, LinearLayer)] == [72, 108]
    T = 6
    for b in range(3):
        xs = [torch.randint(0, 9, (1, 2, 6, 6), generator=g).float() / 8 for _ in range(T)]
        with torch.no_grad():
            ref = _sinabs_run(net, xs)
        ours = _scores(ir, [x.reshape(1, -1) for x in xs])
        assert float_bits(ref) == float_bits(ours)
        art = compile_model(ir, backend="portable_c", num_timesteps=T, input_temporal_mode=TemporalMode.VARYING)
        flat = torch.cat([x.reshape(-1) for x in xs]).tolist()
        assert run_memsum(art.files, tmp_path / f"c{b}", [flat]) == [float_bits(ref[0])]


def test_sinabs_lif_needs_exponential_discretization():
    """Sinabs LIF is v <- a v + (1 - a) I with a = exp(-1/tau); its NIR export is LIF(tau, r = 1). Forward Euler
    (NIR's literal ODE step) gives beta = 1 - 1/tau instead; discretization="exp" reproduces Sinabs to float rounding."""
    _sinabs()
    import sinabs.layers as sl
    from sinabs import nir as snir
    from sinabs.activation import MembraneReset, SingleSpike
    torch.manual_seed(3)
    lif = lambda tau: sl.LIF(tau_mem=tau, spike_fn=SingleSpike, reset_fn=MembraneReset())
    net = torch.nn.Sequential(torch.nn.Linear(6, 8), lif(4.0), torch.nn.Linear(8, 3), lif(8.0))
    graph = snir.to_nir(net, torch.zeros(1, 1, 6))
    xs = [torch.rand(16, 6) * 3 for _ in range(12)]
    with torch.no_grad():
        ref = _sinabs_run(net, [x[:, None, :] for x in xs])
    exp_ir = parse_nir_graph(graph, dt=1.0, discretization="exp")
    euler_ir = parse_nir_graph(graph, dt=1.0)
    assert np.isclose(exp_ir.layers[1].beta[0], np.exp(-0.25)) and euler_ir.layers[1].beta[0] == 0.75
    assert float((_scores(exp_ir, xs) - ref).abs().max()) < 1e-4
    assert float((_scores(euler_ir, xs) - ref).abs().max()) > 1e-2
