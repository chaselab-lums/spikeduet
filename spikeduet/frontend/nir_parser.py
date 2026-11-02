"""NIR frontend: a NIR graph (nir >= 1.0) -> Source IR (`SNNModel`).

The graph must be a single chain from one `Input` to one `Output`. Linear operators between two
neuron nodes (Affine, Linear, Scale, Flatten, Conv1d, Conv2d, AvgPool2d, SumPool2d) are composed into
one dense `LinearLayer`; conv and pooling are lowered to Toeplitz matrices (the same lowering the
ConvSNN frontend uses), so they suit small layers only (`max_dense_weights`). Every neuron node then
becomes a `LIFLayer`. Anything else raises `NIRImportError` naming the node.

Discretization (forward Euler with the caller's `dt`; NIR has no time step of its own). NIR's LIF is
    tau dv/dt = (v_leak - v) + r I
so one step is  v <- beta v + s I + (1 - beta) v_leak  with  beta = 1 - dt/tau,  s = r dt/tau.
`discretization="exp"` instead uses the exact solution for an input held over the step (zero-order hold,
what Sinabs and Norse simulate): beta = exp(-dt/tau), s = r (1 - beta). NIR does not say which one a graph means.
`s` is folded into the preceding weights and bias, `(1 - beta) v_leak` into the bias. A scale within
`SCALE_SNAP` of 1 is taken as exactly 1 (a beta -> tau -> beta round trip rarely returns s == 1.0 to
the last bit). IF is LIF with beta = 1 and s = r. LI (no threshold) is allowed only as the last neuron
node and becomes reset "none" with an unreachable threshold, so the readout sees its membrane.

`semantics` decides what a neuron node means:
  * "nir" (default): the NIR specification only. LIF / IF reset to `v_reset`, which must be 0
    (= the IR's "zero" reset), in the same step as the spike. NIR has no subtract reset, reset delay,
    graded spikes, inhibition or state quantization.
  * "annotated": if the node's `metadata["snntorch"]` holds the snnTorch neuron (beta, threshold,
    reset_mechanism, reset_delay, graded_spikes_factor), that is used verbatim -- a lossless path for
    graphs written by `snntorch_chain_to_nir(..., annotate=True)`. Nodes without it fall back to "nir".

`nonphysical` decides what happens to time constants NIR cannot express a discrete neuron with:
tau = inf (beta = 1 in snnTorch's export divides by zero), tau <= 0 (beta > 1), or a non-finite r.
"error" (default) raises; "clamp" takes beta = min(1 - dt/tau, 1) with tau = inf -> beta = 1, and for
r = tau = inf takes s = 1 (the limit of the snnTorch exporter's r = tau/dt). Every such neuron is
listed in `report["clamped"]`.

`snntorch_chain_to_nir()` writes a [Linear -> Leaky]^N snnTorch model as a NIR graph using snnTorch's
own per-module converter (`snntorch.export_nir._extract_snntorch_module`), only broadcasting scalar
neuron parameters to the layer width so NIR type inference succeeds (the stock `export_to_nir` fails
type inference on scalar-beta models with nir 1.0.8 / nirtorch 2.6).
"""
from typing import Optional

import numpy as np
import torch

from spikeduet.ir.model import LinearLayer, LIFLayer, SNNModel
from spikeduet.ir.verifier import verify_module_ir

SCALE_SNAP = 1e-6
LI_THRESHOLD = 3.0e38                     # float32-finite and never reached by a membrane
_LINEAR_NODES = ("Affine", "Linear", "Scale", "Flatten", "Conv1d", "Conv2d", "AvgPool2d", "SumPool2d")
_NEURON_NODES = ("LIF", "IF", "LI")


class NIRImportError(ValueError):
    """A NIR graph or node the compiler cannot represent exactly; the message names the node."""


def _nir():
    try:
        import nir
    except ImportError as e:                                     # pragma: no cover
        raise NIRImportError("the NIR frontend needs the 'nir' package (pip install nir)") from e
    return nir


def _kind(node) -> str:
    return type(node).__name__


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _io_shape(node, key: str) -> Optional[tuple]:
    t = getattr(node, key, None) or {}
    for v in t.values():
        s = tuple(int(d) for d in np.asarray(v).ravel())
        if s:
            return s
    return None


# ------------------------------------------------------------------ graph walk
def _chain(graph) -> list:
    nir = _nir()
    nodes, edges = graph.nodes, list(graph.edges)
    for name, node in nodes.items():
        if isinstance(node, nir.NIRGraph):
            raise NIRImportError(f"node '{name}': nested NIRGraph (recurrent or composite subgraph) is not supported")
    for s, d in edges:
        for n in (s, d):
            if n not in nodes:
                raise NIRImportError(f"edge {s!r} -> {d!r} names an unknown node {n!r} (subgraph ports are not supported)")
    succ, pred = {}, {}
    for s, d in edges:
        succ.setdefault(s, []).append(d)
        pred.setdefault(d, []).append(s)
    for n, ds in succ.items():
        if len(ds) > 1:
            raise NIRImportError(f"node '{n}' has {len(ds)} consumers {ds}: branching graphs are not supported")
    for n, ss in pred.items():
        if len(ss) > 1:
            raise NIRImportError(f"node '{n}' has {len(ss)} producers {ss}: merging graphs are not supported")
    inputs = [n for n, v in nodes.items() if isinstance(v, nir.Input)]
    outputs = [n for n, v in nodes.items() if isinstance(v, nir.Output)]
    if len(inputs) != 1 or len(outputs) != 1:
        raise NIRImportError(f"need exactly one Input and one Output node, found inputs={inputs} outputs={outputs}")
    order, cur, seen = [], inputs[0], set()
    while True:
        if cur in seen:
            raise NIRImportError(f"cycle through node '{cur}': recurrent graphs are not supported")
        seen.add(cur)
        order.append(cur)
        if cur == outputs[0]:
            break
        nxt = succ.get(cur, [])
        if not nxt:
            raise NIRImportError(f"node '{cur}' has no consumer and is not the Output node")
        cur = nxt[0]
    stray = set(nodes) - seen
    if stray:
        raise NIRImportError(f"nodes not on the Input -> Output chain: {sorted(stray)}")
    return order


# ------------------------------------------------------------------ linear operators as dense maps
def _conv_matrix(w: np.ndarray, in_shape: tuple, stride, padding, dilation, groups) -> tuple:
    """Toeplitz matrix of a conv over a (C, *spatial) input; returns (M [out, in], out_shape)."""
    c_in, sp = in_shape[0], in_shape[1:]
    c_out, cpg, k = w.shape[0], w.shape[1], w.shape[2:]
    nd = len(sp)
    stride, padding, dilation = (tuple(int(v) for v in np.broadcast_to(np.asarray(a).ravel(), (nd,)))
                                 for a in (stride, padding, dilation))
    if c_in != cpg * groups:
        raise NIRImportError(f"conv weight {w.shape} does not match input {in_shape} with groups={groups}")
    out_sp = tuple((sp[i] + 2 * padding[i] - dilation[i] * (k[i] - 1) - 1) // stride[i] + 1 for i in range(nd))
    if min(out_sp) < 1:
        raise NIRImportError(f"conv output is empty for input {in_shape}, kernel {k}")
    M = np.zeros((c_out * int(np.prod(out_sp)), c_in * int(np.prod(sp))))
    in_idx = np.arange(c_in * int(np.prod(sp))).reshape((c_in,) + sp)
    opg = c_out // groups
    for o in range(c_out):
        g = o // opg
        for opos in np.ndindex(*out_sp):
            row = np.ravel_multi_index((o,) + opos, (c_out,) + out_sp)
            for ci in range(cpg):
                for kpos in np.ndindex(*k):
                    ipos = tuple(opos[i] * stride[i] - padding[i] + kpos[i] * dilation[i] for i in range(nd))
                    if all(0 <= ipos[i] < sp[i] for i in range(nd)):
                        M[row, in_idx[(g * cpg + ci,) + ipos]] += w[(o, ci) + kpos]
    return M, (c_out,) + out_sp


def _pool_matrix(in_shape: tuple, kernel, stride, padding, average: bool) -> tuple:
    c, sp = in_shape[0], in_shape[1:]
    nd = len(sp)
    k = tuple(int(v) for v in np.broadcast_to(np.asarray(kernel).ravel(), (nd,)))
    s = tuple(int(v) for v in np.broadcast_to(np.asarray(stride if stride is not None else kernel).ravel(), (nd,)))
    if np.any(np.asarray(padding) != 0):
        raise NIRImportError("pooling with padding is not supported (padding semantics differ between frameworks)")
    out_sp = tuple((sp[i] - k[i]) // s[i] + 1 for i in range(nd))
    M = np.zeros((c * int(np.prod(out_sp)), c * int(np.prod(sp))))
    v = 1.0 / float(np.prod(k)) if average else 1.0
    for ch in range(c):
        for opos in np.ndindex(*out_sp):
            row = np.ravel_multi_index((ch,) + opos, (c,) + out_sp)
            for kpos in np.ndindex(*k):
                ipos = tuple(opos[i] * s[i] + kpos[i] for i in range(nd))
                M[row, np.ravel_multi_index((ch,) + ipos, (c,) + sp)] = v
    return M, (c,) + out_sp


def _linear_map(name: str, node, shape: tuple, max_dense: int) -> tuple:
    """(A [out, in] or None for identity, c [out] or None, out_shape) of one linear operator."""
    kind, n_in = _kind(node), int(np.prod(shape))
    if kind in ("Affine", "Linear"):
        W = _arr(node.weight)
        if W.ndim != 2 or W.shape[1] != n_in:
            raise NIRImportError(f"node '{name}' ({kind}): weight {W.shape} does not take an input of {n_in} values")
        b = _arr(node.bias).ravel() if kind == "Affine" else None
        return W, b, (W.shape[0],)
    if kind == "Scale":
        s = np.broadcast_to(_arr(node.scale), shape).ravel()
        return np.diag(s), None, shape
    if kind == "Flatten":
        return None, None, (n_in,)
    if kind in ("Conv1d", "Conv2d"):
        nd = 1 if kind == "Conv1d" else 2
        if len(shape) == nd + 2 and shape[0] == 1:          # exporters often keep a batch dim of 1 on the Input node
            shape = shape[1:]
        if len(shape) != nd + 1:
            raise NIRImportError(f"node '{name}' ({kind}): input of shape {shape} is not (C, {'L' if nd == 1 else 'H, W'})")
        w = _arr(node.weight)
        out_sz = w.shape[0] * n_in // shape[0]          # upper bound (stride 1, same padding)
        if out_sz * n_in > max_dense:
            raise NIRImportError(f"node '{name}' ({kind}): Toeplitz lowering would need ~{out_sz * n_in:,} weights "
                                 f"(limit {max_dense:,}); large convolutions are not supported")
        if isinstance(node.padding, str):
            raise NIRImportError(f"node '{name}' ({kind}): string padding '{node.padding}' is not supported")
        M, out_shape = _conv_matrix(w, shape, node.stride, node.padding, node.dilation, int(node.groups))
        b = node.bias
        c = None if b is None else np.repeat(np.broadcast_to(_arr(b).ravel(), (w.shape[0],)), int(np.prod(out_shape[1:])))
        return M, c, out_shape
    if kind in ("AvgPool2d", "SumPool2d"):
        if len(shape) == 4 and shape[0] == 1:
            shape = shape[1:]
        if len(shape) != 3:
            raise NIRImportError(f"node '{name}' ({kind}): input of shape {shape} is not (C, H, W)")
        M, out_shape = _pool_matrix(shape, node.kernel_size, node.stride, node.padding, kind == "AvgPool2d")
        return M, None, out_shape
    raise NIRImportError(f"node '{name}': unsupported NIR operator {kind}")


# ------------------------------------------------------------------ neurons
def _per_neuron(name: str, field: str, x, shape: tuple) -> np.ndarray:
    a, n = _arr(x), int(np.prod(shape))
    if a.size == 1:
        return np.full(n, float(a.ravel()[0]))
    if a.size == n:
        return a.ravel()
    try:
        return np.broadcast_to(a, shape).ravel().astype(np.float64)
    except ValueError:
        raise NIRImportError(f"node '{name}': {field} has shape {a.shape}, layer has {n} neurons {shape}") from None


def _compact(v: np.ndarray) -> list:
    return [float(v[0])] if np.all(v == v[0]) else [float(x) for x in v]


def _neuron(name: str, node, shape: tuple, dt: float, semantics: str, nonphysical: str, is_last: bool, report: dict,
            discretization: str = "euler"):
    """(beta, scale s, bias shift, LIFLayer kwargs) for one neuron node."""
    kind = _kind(node)
    meta = (getattr(node, "metadata", None) or {}).get("snntorch") if semantics == "annotated" else None
    if meta is not None:
        report["annotated"].append(name)
        n = int(np.prod(shape))
        beta = _per_neuron(name, "beta", meta["beta"], shape)
        kw = dict(threshold=_compact(_per_neuron(name, "threshold", meta["threshold"], shape)),
                  reset_mechanism=str(meta["reset_mechanism"]), reset_delay=bool(meta["reset_delay"]),
                  graded_spikes_factor=_compact(_per_neuron(name, "graded", meta.get("graded_spikes_factor", 1.0), shape)))
        return np.clip(beta, 0.0, 1.0), np.ones(n), np.zeros(n), kw
    if kind == "LI" and not is_last:
        raise NIRImportError(f"node '{name}': LI (no spikes) is only supported as the last neuron node")
    if kind in ("LIF", "IF"):
        v_reset = _per_neuron(name, "v_reset", node.v_reset, shape)
        if np.any(v_reset != 0):
            raise NIRImportError(f"node '{name}' ({kind}): v_reset != 0 (reset to a nonzero potential) is not supported")
    r = _per_neuron(name, "r", node.r, shape)
    if kind == "IF":
        tau = np.full_like(r, np.inf)
        beta, s = np.ones_like(r), r.copy()
        v_leak = np.zeros_like(r)
        bad = ~np.isfinite(r)
    else:
        tau = _per_neuron(name, "tau", node.tau, shape)
        v_leak = _per_neuron(name, "v_leak", node.v_leak, shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            if discretization == "exp":
                beta = np.exp(-dt / tau)
                s = r * (1.0 - beta)
            else:
                beta = 1.0 - dt / tau
                s = r * dt / tau
        beta = np.where((beta < 0) & (beta > -SCALE_SNAP), 0.0, beta)      # tau == dt stored in float32
        bad = ~np.isfinite(tau) | (tau <= 0) | ~np.isfinite(r) | ~np.isfinite(s) | (beta > 1) | (beta < 0)
        if np.any(bad):
            if nonphysical == "error":
                i = int(np.flatnonzero(bad)[0])
                raise NIRImportError(f"node '{name}' ({kind}): {int(bad.sum())} neurons have a time constant NIR cannot "
                                     f"turn into a discrete neuron (first: tau={tau[i]}, r={r[i]}); "
                                     f"pass nonphysical='clamp' to map them to beta in [0, 1]")
            both_inf = np.isinf(tau) & np.isinf(r)
            s = np.where(both_inf, 1.0, s)
            beta = np.clip(np.where(np.isinf(tau), 1.0, beta), 0.0, 1.0)
            s = np.where(np.isfinite(s), s, 0.0)
            report["clamped"].append({"node": name, "neurons": int(bad.sum())})
    s = np.where(np.abs(s - 1.0) <= SCALE_SNAP, 1.0, s)
    shift = (1.0 - beta) * v_leak
    if kind == "LI":
        kw = dict(threshold=[LI_THRESHOLD], reset_mechanism="none", reset_delay=False)
    else:
        kw = dict(threshold=_compact(_per_neuron(name, "v_threshold", node.v_threshold, shape)),
                  reset_mechanism="zero", reset_delay=False)
    return beta, s, shift, kw


# ------------------------------------------------------------------ public API
def parse_nir_graph(graph, *, dt: float, semantics: str = "nir", nonphysical: str = "error", discretization: str = "euler",
                    input_shape: Optional[tuple] = None, max_dense_weights: int = 4_000_000,
                    report: Optional[dict] = None) -> SNNModel:
    """Parse a chain-shaped NIR graph into verified Source IR. See the module docstring."""
    if discretization not in ("euler", "exp"):
        raise ValueError("discretization must be 'euler' or 'exp'")
    if semantics not in ("nir", "annotated"):
        raise ValueError("semantics must be 'nir' or 'annotated'")
    if nonphysical not in ("error", "clamp"):
        raise ValueError("nonphysical must be 'error' or 'clamp'")
    if not dt > 0:
        raise ValueError("dt must be positive")
    report = {} if report is None else report
    report.update(annotated=[], clamped=[])
    order = _chain(graph)
    nodes = graph.nodes
    neuron_names = [n for n in order if _kind(nodes[n]) in _NEURON_NODES]
    shape = tuple(input_shape) if input_shape else _io_shape(nodes[order[0]], "output_type") or _io_shape(nodes[order[0]], "input_type")
    if not shape:
        raise NIRImportError("the Input node has no shape; pass input_shape=")
    layers, W, b, n_lin = [], None, None, 0
    for name in order[1:-1]:
        node, kind = nodes[name], _kind(nodes[name])
        if kind in _LINEAR_NODES:
            A, c, shape = _linear_map(name, node, shape, max_dense_weights)
            if A is not None:
                W = A if W is None else A @ W
                b = (np.zeros(A.shape[0]) if b is None else A @ b) if (b is not None or c is not None) else None
                if c is not None:
                    b = b + c
            n_lin += kind != "Flatten"
            continue
        if kind not in _NEURON_NODES:
            raise NIRImportError(f"node '{name}': unsupported NIR node type {kind} "
                                 f"(supported: {', '.join(_LINEAR_NODES + _NEURON_NODES)})")
        if n_lin == 0 or W is None:
            raise NIRImportError(f"node '{name}' ({kind}) is not preceded by a linear operator: "
                                 f"the IR needs [Linear -> neuron] pairs")
        beta, s, shift, kw = _neuron(name, node, shape, dt, semantics, nonphysical, name == neuron_names[-1], report, discretization)
        n_out = W.shape[0]
        Ws = (W * s[:, None]).astype(np.float32)
        bias = None
        if b is not None or np.any(shift != 0):
            bias = (np.zeros(n_out) if b is None else b) * s + shift
        lin = LinearLayer(name=f"{name}.affine", in_features=W.shape[1], out_features=n_out,
                          weight=torch.from_numpy(Ws).contiguous(),
                          bias=None if bias is None else torch.from_numpy(bias.astype(np.float32)))
        layers += [lin, LIFLayer(name=name, beta=_compact(beta), **kw)]
        W, b, n_lin = None, None, 0
    if W is not None or n_lin:
        raise NIRImportError("the graph ends with linear operators after the last neuron node; "
                             "the IR needs every Linear to be followed by a neuron")
    if not layers:
        raise NIRImportError("the graph has no neuron nodes")
    model = SNNModel(layers=layers)
    verify_module_ir(model)
    return model


DT_SNNTORCH = 1e-4                        # the dt snnTorch's NIR exporter hard-codes


def snntorch_chain_to_nir(model: torch.nn.Module, *, annotate: bool = False, stock_failed: Optional[list] = None):
    """A [Linear -> Leaky]^N snnTorch model as a NIR graph, using snnTorch's own module converter.

    The chain order comes from the snnTorch frontend (FX trace). Scalar neuron parameters are broadcast
    to the layer width (semantics unchanged; NIR type inference needs it). Where snnTorch's converter
    itself fails (it asserts that beta and threshold have the same shape), its formulas are applied with
    broadcasting and the layer is appended to `stock_failed`. With `annotate`, each LIF node
    also carries `metadata["snntorch"]` = the exact snnTorch neuron, which `parse_nir_graph(...,
    semantics="annotated")` reads back losslessly.
    """
    nir = _nir()
    from snntorch.export_nir import _extract_snntorch_module
    from spikeduet.frontend.snntorch_parser import parse_snntorch_model
    ir = parse_snntorch_model(model)
    nodes, chain = {}, []
    stock_failed = [] if stock_failed is None else stock_failed
    first = ir.layers[0]
    nodes["input"] = nir.Input(input_type=np.array([first.in_features]))
    for lin, lif in zip(ir.layers[0::2], ir.layers[1::2]):
        n = lin.out_features
        mod = model.get_submodule(lif.name)
        with np.errstate(divide="ignore", invalid="ignore"):
            a = _extract_snntorch_module(model.get_submodule(lin.name))
            try:
                l = _extract_snntorch_module(mod)
                if not isinstance(l, nir.LIF):
                    raise NIRImportError(f"'{lif.name}': snnTorch exported {type(l).__name__}, expected LIF")
                raw = {f: getattr(l, f) for f in ("tau", "r", "v_leak", "v_threshold", "v_reset")}
            except AssertionError:
                # snnTorch's converter asserts equal shapes, so a per-neuron beta with a scalar threshold
                # (or the reverse) cannot be exported at all; apply its exact formulas with broadcasting
                beta = mod.beta.detach().numpy()
                tau = DT_SNNTORCH / (1 - beta)
                raw = dict(tau=tau, r=tau / DT_SNNTORCH, v_leak=np.zeros_like(beta),
                           v_threshold=mod.threshold.detach().numpy(), v_reset=np.zeros(1))
                stock_failed.append(lif.name)
        p = {f: np.broadcast_to(np.asarray(v, dtype=np.float32), (n,)).copy() for f, v in raw.items()}
        meta = {}
        if annotate:
            meta = {"snntorch": dict(beta=list(lif.beta), threshold=list(lif.threshold),
                                     reset_mechanism=lif.reset_mechanism, reset_delay=bool(lif.reset_delay),
                                     graded_spikes_factor=list(lif.graded_spikes_factor))}
        l = nir.LIF(**p, metadata=meta)
        nodes[lin.name], nodes[lif.name] = a, l
        chain += [lin.name, lif.name]
    nodes["output"] = nir.Output(output_type=np.array([ir.layers[-2].out_features]))
    names = ["input"] + chain + ["output"]
    with np.errstate(divide="ignore", invalid="ignore"):
        g = nir.NIRGraph(nodes=nodes, edges=list(zip(names[:-1], names[1:])))
        g.infer_types()
    return g
