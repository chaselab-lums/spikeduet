"""Hidden-neuron reordering for the range-sparse MVP realization (opt-in, EFR32 backend support code).

The order of the neurons of a hidden layer is arbitrary: permuting them (the rows of the layer's own Linear, its
per-neuron LIF parameters, and the columns of the next Linear) relabels the network without changing what it
computes. The range-sparse MVP kernel, however, only skips CONTIGUOUS runs of inactive column pairs, so its cost
depends on that order. `cofiring_order` picks an order that places neurons that fire together next to each other,
from a calibration activity trace (never the evaluation split); `permute_hidden_neurons` applies an order to a
source-IR model. Nothing here is target arithmetic: the permuted model is ordinary source IR and goes through the
unchanged verifier / lowering / backends.

Numerics: every hidden neuron computes exactly the same values as before. The NEXT layer sums the same terms in a
different order, so its FP32 results can differ in the last bits (reordered floating-point summation), never more.
"""
import dataclasses
from typing import List, Sequence

import numpy as np

from spikeduet.ir.model import LIFLayer, LinearLayer, SNNModel


def cofiring_order(spikes: np.ndarray) -> List[int]:
    """Order of the N neurons of one layer that clusters co-active neurons: average-linkage hierarchical clustering
    of the Jaccard co-firing distance, with optimal leaf ordering. `spikes`: [K, N] bool/0-1, one row per timestep
    of a calibration trace. Neurons that never fire are appended at the end. Deterministic. Returns `order` with
    new position p holding old neuron order[p]."""
    from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
    from scipy.spatial.distance import squareform
    s = np.asarray(spikes, dtype=np.float32)
    co = s.T @ s
    n = np.diag(co).copy()
    alive = np.flatnonzero(n > 0)
    if len(alive) < 3:
        return list(range(s.shape[1]))
    c = co[np.ix_(alive, alive)]
    d = 1.0 - c / np.maximum(n[alive][:, None] + n[alive][None, :] - c, 1.0)
    np.fill_diagonal(d, 0.0)
    dv = squareform(d, checks=False)
    leaves = leaves_list(optimal_leaf_ordering(linkage(dv, "average"), dv))
    return [int(v) for v in alive[leaves]] + [int(v) for v in np.flatnonzero(n == 0)]


def permute_hidden_neurons(model: SNNModel, hidden_layer: int, order: Sequence[int]) -> SNNModel:
    """A new SNNModel whose hidden layer `hidden_layer` (0 = the first [Linear -> Leaky] pair) has its neurons
    relabelled: new neuron p is old neuron order[p]. The output layer cannot be permuted (class indices)."""
    linears = [i for i, l in enumerate(model.layers) if isinstance(l, LinearLayer)]
    lifs = [i for i, l in enumerate(model.layers) if isinstance(l, LIFLayer)]
    if not 0 <= hidden_layer < len(linears) - 1:
        raise ValueError(f"hidden_layer {hidden_layer} is not a hidden layer of a {len(linears)}-layer model")
    lin, lif, nxt = model.layers[linears[hidden_layer]], model.layers[lifs[hidden_layer]], model.layers[linears[hidden_layer + 1]]
    n = lin.out_features
    order = [int(v) for v in order]
    if sorted(order) != list(range(n)):
        raise ValueError(f"order must be a permutation of 0..{n - 1}")
    if lif.inhibition:
        raise ValueError("an inhibited layer breaks ties by neuron index; reordering it is not supported")

    def per_neuron(values):
        return list(values) if len(values) == 1 else [values[i] for i in order]

    layers = list(model.layers)
    layers[linears[hidden_layer]] = dataclasses.replace(
        lin, weight=lin.weight[order].clone(), bias=None if lin.bias is None else lin.bias[order].clone())
    layers[lifs[hidden_layer]] = dataclasses.replace(
        lif, beta=per_neuron(lif.beta), threshold=per_neuron(lif.threshold), graded_spikes_factor=per_neuron(lif.graded_spikes_factor))
    layers[linears[hidden_layer + 1]] = dataclasses.replace(nxt, weight=nxt.weight[:, order].clone())
    return dataclasses.replace(model, layers=layers)
