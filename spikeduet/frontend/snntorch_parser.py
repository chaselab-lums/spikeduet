"""snnTorch frontend: ``nn.Module`` -> ``SNNModel``.

Two sources of truth are combined:

* a ``torch.fx`` trace, with ``snntorch.Leaky`` kept as a leaf module (tracing into it fails
  on a data-dependent shape check), gives execution order and connectivity;
* the live module instances (``model.get_submodule``) give the parameters.

Only a strict ``[Linear -> Leaky]^N`` chain is accepted. Anything else raises one of the three
exceptions below instead of being approximated. Tracing does not run the model, so no
``Leaky.mem`` buffer is touched.

Supported ``snntorch.Leaky`` configuration: ``beta``, ``threshold`` and ``graded_spikes_factor``
as one shared value or one value per neuron (learned or fixed); ``reset_mechanism`` in
``{"subtract", "zero", "none"}``; either ``reset_delay``; either ``inhibition``; ``state_quant``
disabled or produced by ``snntorch.functional.quant.state_quant(...)``.
"""

import operator
from typing import List

import torch
import torch.fx as fx
import torch.nn as nn
import snntorch as snn

from spikeduet.ir.model import LIFLayer, LinearLayer, SNNModel, StateQuantSpec
from spikeduet.ir.verifier import verify_module_ir


class UnsupportedOperationError(Exception):
    """A leaf module other than ``nn.Linear`` or ``snntorch.Leaky``."""

    def __init__(self, module_name: str, module_type: type) -> None:
        self.module_name = module_name
        self.module_type = module_type
        super().__init__(
            f"Unsupported module '{module_name}' of type "
            f"'{module_type.__module__}.{module_type.__qualname__}'. "
            "Only torch.nn.Linear and snntorch.Leaky leaf modules are "
            "currently supported by the frontend."
        )


class UnsupportedGraphStructureError(Exception):
    """The traced graph is not a single ``[Linear -> Leaky]^N`` chain: wrong ordering,
    branching or merging, module reuse, unsupported calls, or malformed membrane plumbing."""


class UnsupportedLeakyConfigurationError(Exception):
    """A ``snntorch.Leaky`` parameter outside the supported subset (see the module docstring)."""


class _LeakyLeafTracer(fx.Tracer):
    """Keeps ``snntorch.Leaky`` as one opaque ``call_module`` node."""

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, snn.Leaky):
            return True
        return super().is_leaf_module(m, module_qualified_name)


def _parse_linear(name: str, module: nn.Linear) -> LinearLayer:
    weight = module.weight.detach().cpu()
    bias = module.bias.detach().cpu() if module.bias is not None else None
    return LinearLayer(
        name=name,
        in_features=module.in_features,
        out_features=module.out_features,
        weight=weight,
        bias=bias,
    )


_SUPPORTED_RESET_MECHANISMS = ("subtract", "zero", "none")

# ``state_quant(...)`` returns a closure ``inner`` capturing a ``levels`` tensor. snnTorch exposes no
# marker class, so a quantizer is recognized by the closure's qualified name and module.
_STATE_QUANT_QUALNAME = "state_quant.<locals>.inner"
_STATE_QUANT_MODULE = "snntorch.functional.quant"


def _extract_state_quant_levels(name: str, fn) -> "torch.Tensor":
    """Returns the ``levels`` tensor captured by a ``state_quant(...)`` closure; any other
    callable raises ``UnsupportedLeakyConfigurationError``."""
    if not callable(fn):
        raise UnsupportedLeakyConfigurationError(
            f"Leaky module '{name}' has state_quant={fn!r}, which is not "
            "callable. only state_quant=False/None "
            "(disabled) or a quantizer produced by "
            "snntorch.functional.quant.state_quant(...)."
        )
    if (
        getattr(fn, "__qualname__", None) != _STATE_QUANT_QUALNAME
        or getattr(fn, "__module__", None) != _STATE_QUANT_MODULE
    ):
        raise UnsupportedLeakyConfigurationError(
            f"Unsupported Leaky.state_quant callable on module '{name}' "
            f"({fn!r}). Only quantizers produced by "
            "snntorch.functional.quant.state_quant(...) are currently "
            "supported -- an arbitrary Python callable (e.g. a lambda or "
            "hand-written quantization function) cannot be compiled, "
            "since quantization is compiled as a resolved "
            "level table, not an executable Python function."
        )

    code = getattr(fn, "__code__", None)
    closure = getattr(fn, "__closure__", None)
    if code is None or closure is None or "levels" not in code.co_freevars:
        raise UnsupportedLeakyConfigurationError(
            f"Leaky module '{name}' has a state_quant callable that looks "
            "like snntorch.functional.quant.state_quant(...)'s own closure "
            "but does not expose the expected 'levels' captured variable "
            "-- refusing to guess at its semantics rather than silently "
            "misinterpreting it."
        )
    levels = closure[code.co_freevars.index("levels")].cell_contents
    if not torch.is_tensor(levels) or levels.dim() != 1 or levels.numel() == 0:
        raise UnsupportedLeakyConfigurationError(
            f"Leaky module '{name}' has a state_quant callable whose "
            f"captured 'levels' is not a non-empty 1-D tensor ({levels!r}) "
            "-- refusing to guess at its semantics."
        )
    return levels


def _parse_leaky_param(name: str, param_name: str, value: torch.Tensor, num_neurons: int) -> List[float]:
    """Accepts one shared value or exactly one value per neuron. General broadcasting shapes
    are rejected because they have no unambiguous per-neuron meaning."""
    numel = value.numel()
    if numel not in (1, num_neurons):
        raise UnsupportedLeakyConfigurationError(
            f"Leaky module '{name}' has {param_name} of shape {tuple(value.shape)} "
            f"({numel} element(s)); only a single scalar "
            f"(1 element) or exactly one value per neuron ({num_neurons} elements "
            f"for this layer's {num_neurons} neurons) -- not arbitrary broadcastable "
            "shapes."
        )
    return value.reshape(-1).tolist()


def _parse_leaky(name: str, module: "snn.Leaky", num_neurons: int) -> LIFLayer:
    if module.state_quant in (False, None):
        state_quant_spec = None
    else:
        levels_tensor = _extract_state_quant_levels(name, module.state_quant)
        state_quant_spec = StateQuantSpec(levels=tuple(levels_tensor.tolist()))

    if module.reset_mechanism not in _SUPPORTED_RESET_MECHANISMS:
        raise UnsupportedLeakyConfigurationError(
            f"Leaky module '{name}' uses reset_mechanism={module.reset_mechanism!r}; "
            f"only {_SUPPORTED_RESET_MECHANISMS!r} (snnTorch "
            "itself only accepts these three -- this should be unreachable via a "
            "validly-constructed Leaky module)."
        )
    reset_delay = bool(module.reset_delay)

    # snnTorch clamps beta to [0, 1] on every forward call. The deployed beta is frozen and clamp is
    # idempotent, so clamping once here is equivalent and saves the runtime a clamp per step.
    beta_values = _parse_leaky_param(name, "beta", module.beta.clamp(0, 1), num_neurons)
    threshold_values = _parse_leaky_param(name, "threshold", module.threshold, num_neurons)

    graded_spikes_values = _parse_leaky_param(
        name, "graded_spikes_factor", module.graded_spikes_factor, num_neurons
    )

    inhibition = bool(module.inhibition)

    return LIFLayer(
        name=name,
        beta=beta_values,
        threshold=threshold_values,
        reset_mechanism=module.reset_mechanism,
        reset_delay=reset_delay,
        graded_spikes_factor=graded_spikes_values,
        inhibition=inhibition,
        state_quant=state_quant_spec,
    )


def parse_snntorch_model(model: nn.Module) -> SNNModel:
    """Parses ``model`` into an ``SNNModel`` with layers in execution order.

    ``model.forward`` must call each ``Linear`` once and each ``Leaky`` as
    ``spk, mem = lif(cur, lif.mem)``, chaining spikes into the next ``Linear``.

    Raises:
        UnsupportedOperationError: a module other than ``nn.Linear``/``snntorch.Leaky``.
        UnsupportedGraphStructureError: the graph is not a single ``[Linear -> Leaky]^N`` chain.
        UnsupportedLeakyConfigurationError: a ``Leaky`` parameter outside the supported subset.
        IRVerificationError: the resulting IR fails ``verify_module_ir``.
    """
    tracer = _LeakyLeafTracer()
    graph = tracer.trace(model)
    all_nodes = list(graph.nodes)

    placeholders = [n for n in all_nodes if n.op == "placeholder"]
    if len(placeholders) != 1:
        raise UnsupportedGraphStructureError(
            f"expected exactly one model input, found {len(placeholders)}"
        )
    outputs = [n for n in all_nodes if n.op == "output"]
    if len(outputs) != 1:
        raise UnsupportedGraphStructureError("expected exactly one graph output")
    output_node = outputs[0]

    visited = {placeholders[0], output_node}
    seen_targets = set()
    layers = []

    current = placeholders[0]
    expect = "linear"  # next expected semantic layer kind

    while True:
        users = list(current.users)
        if len(users) != 1:
            raise UnsupportedGraphStructureError(
                f"node '{current.name}' has {len(users)} consumer(s); the "
                "frontend requires a single sequential chain with no "
                "branching or merging"
            )
        node = users[0]
        if node is output_node:
            break

        if node.op != "call_module":
            raise UnsupportedGraphStructureError(
                f"unexpected {node.op} node '{node.name}' (target="
                f"{node.target!r}) following '{current.name}'; only "
                "Linear/Leaky module calls are supported in the semantic "
                "chain (no other call_function, call_method, or get_attr "
                "is permitted here)"
            )

        target = node.target
        if target in seen_targets:
            raise UnsupportedGraphStructureError(
                f"module '{target}' (node '{node.name}') is used more than "
                "once in the forward graph; the frontend does not "
                "support semantic module reuse"
            )
        module = model.get_submodule(target)

        if expect == "linear":
            if not isinstance(module, nn.Linear):
                if isinstance(module, snn.Leaky):
                    raise UnsupportedGraphStructureError(
                        f"expected a Linear layer at node '{node.name}' "
                        f"(module '{target}') but found a Leaky layer; the frontend "
                        "only supports the strict [Linear -> Leaky]^N "
                        "ordering"
                    )
                raise UnsupportedOperationError(target, type(module))
            if len(node.args) != 1 or node.args[0] is not current or node.kwargs:
                raise UnsupportedGraphStructureError(
                    f"Linear module '{target}' at node '{node.name}' is not "
                    "called directly on the preceding layer's output; "
                    "branching/merging into a Linear layer is not supported"
                )
            seen_targets.add(target)
            visited.add(node)
            layers.append(_parse_linear(target, module))
            current = node
            expect = "leaky"
            continue

        # expect == "leaky"
        if not isinstance(module, snn.Leaky):
            if isinstance(module, nn.Linear):
                raise UnsupportedGraphStructureError(
                    f"expected a Leaky layer at node '{node.name}' (module "
                    f"'{target}') but found another Linear layer; only "
                    "supports the strict [Linear -> Leaky]^N ordering"
                )
            raise UnsupportedOperationError(target, type(module))

        if len(node.args) != 2 or node.kwargs:
            raise UnsupportedGraphStructureError(
                f"Leaky module '{target}' at node '{node.name}' is not "
                "called with the expected (current, mem) signature"
            )
        cur_arg, mem_arg = node.args
        if cur_arg is not current:
            raise UnsupportedGraphStructureError(
                f"Leaky module '{target}' at node '{node.name}' does not "
                "consume the immediately preceding layer's output as its "
                "current input"
            )
        expected_mem_target = f"{target}.mem"
        if (
            not isinstance(mem_arg, fx.Node)
            or mem_arg.op != "get_attr"
            or mem_arg.target != expected_mem_target
        ):
            raise UnsupportedGraphStructureError(
                f"Leaky module '{target}' at node '{node.name}' has an "
                f"unexpected membrane-state input ({mem_arg!r}); only that "
                f"same module's own '{expected_mem_target}' buffer is "
                "supported"
            )
        if len(list(mem_arg.users)) != 1:
            raise UnsupportedGraphStructureError(
                f"get_attr node '{mem_arg.name}' ('{expected_mem_target}') "
                "is used more than once; membrane-state plumbing must be "
                "single-use"
            )

        seen_targets.add(target)
        visited.add(node)
        visited.add(mem_arg)

        leaky_users = list(node.users)
        if len(leaky_users) != 2:
            raise UnsupportedGraphStructureError(
                f"Leaky module '{target}' at node '{node.name}' has "
                f"{len(leaky_users)} consumer(s); its (spk, mem) tuple must "
                "be unpacked by exactly two getitem operations"
            )
        getitem_by_index = {}
        for user in leaky_users:
            if user.op != "call_function" or user.target is not operator.getitem:
                raise UnsupportedGraphStructureError(
                    f"node '{user.name}' ({user.op}, target={user.target!r}) "
                    f"consumes Leaky module '{target}' but is not a "
                    "supported tuple-unpacking getitem operation"
                )
            if len(user.args) != 2 or user.args[0] is not node or user.kwargs:
                raise UnsupportedGraphStructureError(
                    f"malformed tuple extraction at node '{user.name}' for "
                    f"Leaky module '{target}'"
                )
            index = user.args[1]
            if index not in (0, 1) or index in getitem_by_index:
                raise UnsupportedGraphStructureError(
                    f"unsupported tuple index {index!r} extracted from "
                    f"Leaky module '{target}' at node '{user.name}'; only "
                    "indices 0 (spk) and 1 (mem) are supported, each "
                    "exactly once"
                )
            getitem_by_index[index] = user
            visited.add(user)

        spk_node, mem_node = getitem_by_index[0], getitem_by_index[1]
        if any(u is not output_node for u in mem_node.users):
            raise UnsupportedGraphStructureError(
                f"the membrane output of Leaky module '{target}' (node "
                f"'{mem_node.name}') feeds further computation; the frontend does "
                "not support using membrane potential as an input elsewhere"
            )

        layers.append(_parse_leaky(target, module, num_neurons=layers[-1].out_features))
        current = spk_node
        expect = "linear"

    if expect != "linear":
        raise UnsupportedGraphStructureError(
            "the model's semantic graph must end with a Leaky layer to "
            "complete a [Linear -> Leaky] block; found a trailing, "
            "unpaired Linear layer"
        )
    if not layers:
        raise UnsupportedGraphStructureError(
            "no supported [Linear -> Leaky] layers were found in the "
            "traced graph"
        )

    leftover = [n for n in all_nodes if n not in visited]
    if leftover:
        n = leftover[0]
        raise UnsupportedGraphStructureError(
            f"node '{n.name}' ({n.op}, target={n.target!r}) is not part of "
            "the supported sequential [Linear -> Leaky]^N chain"
        )

    result = SNNModel(layers=layers)

    verify_module_ir(result)
    return result
