# SpikeDuet

**Compile stateful snnTorch spiking neural networks to C for microcontrollers.**

The name refers to the EFR32 target: the Cortex-M33 and the MVP matrix accelerator play together, and the compiler decides which of the two runs each layer.

SpikeDuet takes a trained `snntorch` model, turns it into a verified intermediate representation, optimizes its timestep schedule, and generates C code for one of two backends:

| Backend | Output | Dependencies |
|---|---|---|
| `portable_c` | Plain C99 that builds with gcc or clang on any target | none |
| `efr32` | Firmware sources for Silicon Labs EFR32 Series-2 parts. Dense matrix work runs on the MVP accelerator or on the Cortex-M33; LIF state runs on the Cortex-M33 | Simplicity SDK (`sl_math_mvp`) |

Every generated build reproduces the Python reference simulator, and the simulator reproduces `snntorch.Leaky` (`snntorch==1.0.0`) exactly.

```
snnTorch nn.Module
   │  frontend (torch.fx trace + live module parameters)
   ▼
Source IR  (SNNModel: [Linear -> LIF]^N)  ──►  reference simulator (ground truth)
   │  verifier, lowering
   ▼
Execution IR  (buffers, timestep loop, preheader)
   │  passes: LICM, spike-staging fusion
   ▼
Backend  ──►  portable C99  |  EFR32 (CPU / MVP / auto placement)
```

---

## Contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Writing a compatible model](#writing-a-compatible-model)
4. [Compiler API](#compiler-api)
5. [Portable-C backend](#portable-c-backend)
6. [EFR32 backend](#efr32-backend)
7. [Firmware runtime and board protocol](#firmware-runtime-and-board-protocol)
8. [Hardware-in-the-loop tools](#hardware-in-the-loop-tools)
9. [ANN baselines](#ann-baselines)
10. [Testing](#testing)
11. [Repository layout](#repository-layout)
12. [Contributing](#contributing)
13. [Limitations](#limitations)
14. [License](#license)

---

## Installation

Python 3.9 or newer. The compiler needs PyTorch, snnTorch, NumPy and SciPy (SciPy only for hidden-neuron reordering).

```bash
git clone <repo-url> SpikeDuet
cd SpikeDuet
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # compiler only
pip install -r requirements-dev.txt  # plus pytest and pyserial (tests, board tools)
pip install -e .                   # optional: import spikeduet from anywhere
```

`snntorch` is pinned to `1.0.0` because the neuron semantics are verified against that exact release.

Optional tools:

- **A host C compiler** (`gcc` or `clang`). Used by the example and by the tests that compile and run generated code; those tests are skipped without one.
- **For EFR32 builds:** Simplicity Studio 5 or the Simplicity SDK, the GNU Arm toolchain, `slc-cli` and Simplicity Commander (see [Building on the board](#building-on-the-board)).

## Quick start

```bash
python examples/quickstart.py
```

The example builds a two-layer SNN and compiles it to portable C. It then builds the C on the host and checks that its predictions match the Python reference. The core of it:

```python
from spikeduet import parse_snntorch_model, compile_model, TemporalMode

ir = parse_snntorch_model(model)                     # nn.Module -> verified source IR
artifact = compile_model(
    ir,
    backend="portable_c",
    num_timesteps=20,
    input_temporal_mode=TemporalMode.VARYING,        # one input vector per timestep
    pipeline="optimized",
)
artifact.write("build/")                             # model_data.h, model_config.h, portable_runtime.{h,c}
```

Generate EFR32 firmware sources for the same model:

```python
artifact = compile_model(ir, backend="efr32", num_timesteps=20,
                         input_temporal_mode=TemporalMode.VARYING,
                         pipeline="optimized", engine="mvp", sparse_mvp=True)
# files: model_data.h, snn_generated_mvp.c  -> copy into the firmware project as model_data.h / snn.c
```

## Writing a compatible model

The frontend accepts one topology: a chain of `nn.Linear` layers, each followed by one `snntorch.Leaky`.

```
[Linear -> Leaky]^N,  N >= 1
```

`forward()` must run **one timestep** and pass each `Leaky` its own membrane buffer:

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.lif1 = nn.Linear(700, 64), snn.Leaky(beta=0.95, reset_delay=False)
        self.fc2, self.lif2 = nn.Linear(64, 20), snn.Leaky(beta=0.95, reset_delay=False)

    def forward(self, x):
        spk1, _ = self.lif1(self.fc1(x), self.lif1.mem)
        spk2, _ = self.lif2(self.fc2(spk1), self.lif2.mem)
        return spk2
```

Your training loop over timesteps can be anything. The compiler only traces this single-step `forward()`.

**Supported `snntorch.Leaky` options**

| Option | Supported values |
|---|---|
| `beta`, `threshold`, `graded_spikes_factor` | One shared value, or exactly one value per neuron. Learned or fixed |
| `reset_mechanism` | `"subtract"`, `"zero"`, `"none"` |
| `reset_delay` | `True` or `False` |
| `inhibition` | `True` (layer-wide winner-take-all) or `False` |
| `state_quant` | disabled, or a quantizer from `snntorch.functional.quant.state_quant(...)` |
| `nn.Linear` bias | with or without |

`beta` is clamped to `[0, 1]` once at export, as snnTorch does on every step.

Unsupported graphs and options raise an error; they are never approximated:

- `UnsupportedOperationError`: any other module type.
- `UnsupportedGraphStructureError`: branching, module reuse, other function calls, wrong ordering.
- `UnsupportedLeakyConfigurationError`: an option outside the table above, such as an arbitrary `state_quant` callable.

## Compiler API

### `compile_model`

```python
compile_model(source_ir, *, backend, num_timesteps=None,
              input_temporal_mode=TemporalMode.VARYING, pipeline=None, **backend_kwargs)
```

| Argument | Meaning |
|---|---|
| `backend` | `"portable_c"` or `"efr32"` |
| `num_timesteps` | Timestep count of the inference loop |
| `input_temporal_mode` | `TemporalMode.INVARIANT` if one input vector is held for all steps (e.g. a static image). `TemporalMode.VARYING` if each step has its own vector (e.g. an event stream). It decides what LICM may hoist and how the generated code reads its input |
| `pipeline` | A name from the table below, or a `PassManager`. Default `"baseline"` |
| `**backend_kwargs` | Backend options (see each backend) |

Returns a `GeneratedArtifact`. `.files` maps file names to text; `.write(dir)` saves them.

### Pipelines

| Name | Passes | Effect |
|---|---|---|
| `baseline` | verify | The naive schedule: every op runs every timestep |
| `licm` | verify, `InvariantAffineHoistingPass`, verify | Affines whose input never changes move before the loop. With an `INVARIANT` input the first layer's affine runs once instead of T times. With a `VARYING` input nothing is hoisted |
| `fusion` | verify, `SpikeStagingFusionPass`, verify | Each LIF writes its spikes straight into the next layer's input buffer, removing a copy per layer and step |
| `optimized` | LICM, then fusion | Both |

All pipelines produce numerically identical results. Lower-level entry points:

```python
from spikeduet.ir.pass_manager import compile_source_to_execution_ir, get_pipeline
from spikeduet.ir.execution_pretty import format_execution_ir

module = compile_source_to_execution_ir(ir, num_timesteps=10,
                                        input_temporal_mode=TemporalMode.INVARIANT,
                                        pipeline=get_pipeline("optimized"))
print(format_execution_ir(module))        # buffers, preheader, loop body
```

### Reference simulator

```python
from spikeduet import run_sequence
outputs, final_state, traces = run_sequence(ir, [x_t for x_t in inputs])   # each x_t: (batch, input_dim)
prediction = sum(t.mem[-1] for t in traces).argmax(dim=1)
```

Every backend reads out the same way: it sums the final layer's membrane over all timesteps and takes the argmax.

## Portable-C backend

```python
compile_model(ir, backend="portable_c", num_timesteps=T, include_harness=False)
```

| File | Contents |
|---|---|
| `model_data.h` | Weights, biases, LIF parameters and the `ISCAS_LAYERS[]` descriptor table |
| `model_config.h` | `ISCAS_STEPS`, `ISCAS_INPUT_TEMPORAL`, `ISCAS_MAX_LAYER_WIDTH` |
| `portable_runtime.h` | LIF helpers and the public API |
| `portable_runtime.c` | The model's schedule. Preheader ops run once and loop ops run every step; all buffers are static |
| `main.c` | Only with `include_harness=True`: a host program that reads samples on stdin and writes predicted classes |

```c
#include "portable_runtime.h"
iscas_model_init();
int cls = iscas_model_run(input);   /* input: [INPUT_DIM], or [STEPS][INPUT_DIM] if ISCAS_INPUT_TEMPORAL */
```

`iscas_model_debug_{affine,lif,convert}_exec_count(layer)` report how often each op ran in the last call. Build it with `-DISCAS_PORTABLE_EXPOSE_MEMSUM` to also export `iscas_model_last_mem_sum()`.

## EFR32 backend

The `efr32` backend generates sources for Silicon Labs EFR32 Series-2 parts with the MVP (Matrix Vector Processor). It was developed on the EFR32MG24 (xG24-DK2601B) and also runs on the EFM32PG28.

### Engines

| `engine=` | Generated files | Meaning |
|---|---|---|
| `None` (default) | `model_data.h` | Descriptor table only, for the hand-written generic runtime |
| `"cpu"` | `model_data.h`, `snn_generated_cpu.c` | Everything on the Cortex-M33 in FP32 |
| `"mvp"` | `model_data.h`, `snn_generated_mvp.c` | Affine layers on the MVP in FP16; LIF updates on the Cortex-M33 |
| `"auto"` | as `"cpu"` or `"mvp"`, plus `placement_report.json` | Chooses each layer's engine from a calibration activity profile (see [Automatic placement](#automatic-placement)) |

Rename the generated engine to `snn.c` in the firmware project. It implements `snn_init()`, `run_snn_inference()` and the streaming functions `iscas_infer_reset()`, `iscas_infer_step()` and `iscas_infer_readout()` that the runtime calls.

### Options

All options are off by default. With every option off the output is the plain dense schedule of the chosen pipeline.

**Sparsity**

| Option | Engine | What it does |
|---|---|---|
| `sparse_mvp=True` | mvp | **Range-sparse MVP.** For each timestep, active input columns are merged into contiguous ranges and only those ranges are sent to the MVP. Results are bit-identical to the dense kernel. Applies to loop-resident layers whose dense work is large enough. A built-in cost-model guard falls back to one full launch on steps where sparsity would not pay |
| `sparse_cpu=True` | cpu | **Event-driven Cortex-M33.** Each affine scans its input for active entries and accumulates only those weight columns (transposed FP32 weights, 16-row register blocks) |
| `cpu_sparse_layers={i, ...}` | mvp | Hybrid build: these layers run event-driven on the Cortex-M33, the rest on the MVP |
| `sparse_activity={layer: (launches, pairs)}` | mvp | Measured per-step activity, used to decide whether hidden spike edges go range-sparse |
| `force_sparse_mvp_layers={i}` | mvp | Make these hidden layers range-sparse regardless of the activity gate |
| `sparse_mvp_layers={i}` | mvp | Restrict range-sparse to these layers |
| `sparse_merge_gap={layer: pairs}` | mvp | Override the gap below which two active ranges are merged. The default comes from the launch-cost model |

**Memory placement**

| Option | Engine | What it does |
|---|---|---|
| `cpu_sparse_weight_placement={layer: "sram" \| "flash"}` | cpu, mvp | Per-layer residency of the transposed weights of event-driven layers. Without this, the `ISCAS_CPU_SPARSE_WT_SRAM` build define decides globally |
| `cpu_sparse_pinned_columns={layer: [cols]}` | cpu, mvp | For a flash-resident layer, copy these frequently used columns to SRAM once at init (select them with `efr32.cpu_sparse.select_hot_columns`) |
| `mvp_flash_weight_layers={i}` | mvp | Keep these layers' FP16 weights in flash and let the MVP read them in place, instead of copying them to SRAM. Needed when the weights do not fit SRAM |
| `mvp_static_io_layers={i}` | mvp | Place these layers' FP16 input/output vectors in static memory, away from the weights. Small MVP layers run up to ~2x faster this way |

**Schedule transforms**

| Option | Engine | What it does |
|---|---|---|
| `fold_output_affine=True` | cpu, mvp (M33 output layer) | Runs the output affine once per inference on the time-summed input instead of once per step. Legal only when the output LIF has a fixed `beta` of 0 and `reset_mechanism="none"`. `output_fold_plan(module)` explains why when it is not legal |
| `dense_window=W` | mvp | Batch W timesteps of the dense input layer into one MVP program (bit-exact) |
| `dense_lean=True` | mvp | Leanest dense input-layer realization |
| `mvp_full_width_layers={i}` | mvp | Dense layer through one raw full-width launch instead of the SDK call |

**Asynchronous execution (experimental)**

| Option | Engine | What it does |
|---|---|---|
| `async_mvp=True` | mvp | Overlaps the Cortex-M33 and the MVP. The MVP runs this step's range-sparse layer while the CPU finishes the previous step's downstream work. Outputs are bit-identical to the blocking build. Needs exactly one range-sparse MVP layer (see `efr32.async_mvp.async_mvp_plan`) |
| `async_poll_neurons=N` | mvp | Completion-poll interval inside the deferred work (default 32 neurons) |
| `async_fast_wait=True` | mvp | Register-only wait loop |
| `async_poll_m33=True` | mvp | Also poll between the blocks of deferred Cortex-M33 event kernels |
| `lif_params_sram={i}` | cpu, mvp | Read these layers' LIF parameters from SRAM copies. Measured slower on the reference board; kept for experiments |

**Instrumentation**

| Option | Engine | What it does |
|---|---|---|
| `stage_profile=True` | cpu, mvp | Emit per-stage DWT cycle markers. They compile to nothing unless the build defines `ISCAS_STAGE_PROFILE` |

### Automatic placement

`engine="auto"` estimates each loop-resident affine's cost on the dense MVP, the range-sparse MVP and the event-driven Cortex-M33, and picks the cheapest. The estimates are cycle-cost models calibrated on silicon, and the planner checks an SRAM estimate against the target's capacity. Activity comes from a calibration profile computed on held-out data:

```python
from spikeduet.backend.efr32 import Efr32Backend, profile_activity, plan_engine_placement, format_placement_report

profile = profile_activity(ir, calibration_inputs, source="validation split")   # inputs: [N, T, C]
profile.save("profile.json")

plan = plan_engine_placement(module, profile)
print(format_placement_report(plan))            # per-layer candidates, choice, runner-up, margin
files = Efr32Backend().generate(module, engine="auto", activity_profile=profile).files
print(plan.required_defines)                    # add these to the firmware build
```

| Argument | Meaning |
|---|---|
| `force_mvp_layers=` / `cpu_sparse_layers=` | Pin layers to the MVP or the Cortex-M33 |
| `placement_platform=` | A `PlatformSpec` with a different SRAM budget |
| `m33_weights=` | `"auto"`, `"sram"` or `"flash"` |

Without a profile, `auto` behaves like `engine="mvp", sparse_mvp=True`.

`plan_with_reordering()` additionally considers relabelling hidden neurons so that neurons that fire together are adjacent. The relabelling is exact and does not change the network's function. It can make a hidden layer's spikes contiguous enough for the range-sparse MVP. Build the candidates with `profile_reordering()`; `efr32.reorder` holds the primitives.

### Build defines

Set these in the firmware project (`-D...`). All are off by default.

| Define | Effect |
|---|---|
| `ISCAS_CPU_SPARSE_WT_SRAM=1` | Copy all transposed Cortex-M33 weights to SRAM at init (else read from flash) |
| `ISCAS_SPARSE_VERIFY` | Recompute every sparse or folded step with the dense kernel and count bitwise mismatches |
| `ISCAS_STAGE_PROFILE` | Activate `stage_profile` markers; prints `STAGEPROF:` after each prediction |
| `ISCAS_COUNT_SPARSE_GEN` | Print MVP launch and pair counters plus raw cycles after each prediction |
| `ISCAS_COUNT_CPU_SPARSE_GEN` | Print event counts of Cortex-M33 event-driven layers |
| `ISCAS_COUNT_EXEC_GEN` | Print per-op execution counts |
| `ISCAS_REPORT_RAWCYCLES` | Print raw DWT cycles and the clock frequency |
| `ISCAS_REPORT_MEMINFO` | Print peak heap and stack use |
| `ISCAS_REPORT_SCORES` | Print the exact readout scores as IEEE-754 bit patterns |
| `ISCAS_ASYNC_NO_OVERLAP`, `ISCAS_ASYNC_PERF` | Async builds: disable the overlap (control build), or print MVP run/stall counters |
| `ISCAS_ENERGY_REPLAY` | Replay mode for energy measurement with an external power monitor |

All reports are printed after the `PRED` line, outside the timed region, so they never change measured latency.

### Cost and energy models

The planner's cycle-cost functions are reusable on their own, for quick estimates before building anything:

- MVP: `mvp_dense_affine_cycles`, `mvp_dense_hidden_affine_cycles`, `mvp_sparse_affine_cycles`, `mvp_sparse_hidden_affine_cycles` (in `efr32.cost_model`).
- Cortex-M33: `efr32.cpu_sparse.cpu_sparse_affine_cycles`.
- Experimental extras: `efr32.async_placement` (a CPU/MVP overlap model) and `efr32.energy` (energy per inference from board power states measured on an EFM32PG28 Pro Kit).

## Firmware runtime and board protocol

`runtime/efr32/` holds the device-side shell that every EFR32 build uses:

| File | Role |
|---|---|
| `main.c`, `app.c`, `app.h` | UART loop, DWT timing, input parsing. Topology-independent |
| `snn.h` | Shared LIF helpers (`iscas_lif_update()` and friends) and the engine contract |

### Building on the board

1. Create a Simplicity Studio project for your board. An empty C project with the `iostream` (USART/VCOM) component and the `math_mvp` component is enough.
2. Copy `runtime/efr32/{main.c,app.c,app.h,snn.h}`, the generated `model_data.h` and the generated engine (renamed `snn.c`) into it.
3. Build with Simplicity Studio, or from the command line:
   ```bash
   slc generate -p <project>.slcp -s <simplicity_sdk> -o makefile -tlcn gcc
   ```
   ```bash
   make -f <project>.Makefile ARM_GCC_DIR=<gnu_arm> SDK_PATH=<simplicity_sdk>
   ```
4. Flash with Simplicity Commander:
   ```bash
   commander flash build/<project>.hex --device <part>
   ```

### UART protocol

All values cross the wire as integers scaled by 10000.

- **Static input** (`ISCAS_INPUT_TEMPORAL` = 0). The board announces `MCU_READY`. The host then sends `ISCAS_INPUT_DIM` whitespace-separated integers per inference. The board answers `PRED:<class>:<infer_us>:<parse_us>:<total_us>`.
- **Temporal input** (`ISCAS_INPUT_TEMPORAL` = 1). Each step is acknowledged before the next is sent:
  1. host: `BEGIN_TEMPORAL <steps> <dim>`; board: `READY_T:0`
  2. per step t: host sends one vector; board: `ACK_T:<t>`
  3. after the last step, the board sends `PRED:...`

If a byte does not arrive within 2 s, the board answers `ERR:UART_TIMEOUT:...` and accepts the next request without a reset.

`infer_us` is measured with DWT cycles around the model computation only, so it excludes UART time.

## Hardware-in-the-loop tools

`tools/hil/` (needs `pyserial`):

```python
from tools.hil.protocol import open_and_sync, scale_vector, send_and_read, run_temporal
from tools.hil.communicator import run_vectors, run_temporal_vectors

ser = open_and_sync("COM8")                                   # waits for MCU_READY
reply = run_temporal(ser, [scale_vector(frame) for frame in frames])
print(reply.predicted_class, reply.infer_us)
```

`tools/hil/firmware_build.py` clones an existing Simplicity Studio project with new generated sources and build defines. It then builds it, failing on any warning, and reports flash use and static SRAM. Requested heap is measured by running `snn_init()` on the host. Set these environment variables for your install:

| Variable | Meaning |
|---|---|
| `SPIKEDUET_SSV5_WORKSPACE` | Simplicity Studio workspace |
| `SPIKEDUET_ARM_GCC_DIR` | GNU Arm toolchain directory |
| `SPIKEDUET_SDK_PATH` | Simplicity SDK |
| `SPIKEDUET_SLT_ARCHIVE` | `slc-cli` install archive |
| `SPIKEDUET_COMMANDER` | `commander` executable |
| `SPIKEDUET_BUILD_PATH` | Extra `PATH` entries for `make`, `sed` and `uname` |

`tools/hil/aem.py` records energy with the Advanced Energy Monitor of an AEM-equipped kit.

## ANN baselines

`spikeduet.baselines.mlp` generates matching firmware for plain MLPs (`Linear -> ReLU -> ...`), for fair SNN-versus-ANN comparisons on the same runtime. It is independent of the SNN IR.

```python
from spikeduet.baselines.mlp import MLPWeights, generate_mlp_firmware

mlp = MLPWeights.from_linears([model.fc1, model.fc2, model.fc3])   # or MLPWeights.from_state_dict_file(...)
files = generate_mlp_firmware(mlp, engine="mvp", num_steps=51,     # model_data.h, snn.h, snn.c
                              event_input_layer=True, reuse_empty_frames=True,
                              static_io_layers=(1, 2), fold_output_affine=True)
```

`num_steps=None` generates a static engine (one forward pass per inference); `engine="cpu"` runs everything on the Cortex-M33.

The temporal generator offers the same optimizations as the SNN backend:

| Option | What it does |
|---|---|
| `event_input_layer` | Event-driven first layer on the Cortex-M33 |
| `reuse_empty_frames` | Skip all-zero frames exactly (a stateless MLP's output for an empty frame is a constant) |
| `static_io_layers` | Static FP16 input/output vectors for the MVP |
| `mvp_flash_weight_layers` | FP16 weights read by the MVP from flash |
| `mvp_split_rows` | Split a layer's weights between SRAM and flash |
| `fold_output_affine` | Output affine once per inference |
| `stage_profile` | Per-stage DWT markers |

## Testing

```bash
pytest
```

The suite covers four groups:

- **Frontend:** checked against live snnTorch, including every reset mode, per-neuron parameters, graded spikes, inhibition and state quantization.
- **IR:** verifiers, passes and analyses.
- **Code generation:** structural checks of the generated text.
- **Execution:** tests that compile and run the generated C on the host and compare its predictions and membrane sums with the reference. These need `gcc` or `clang` and are skipped otherwise.

To check that a refactor of a code generator changes no generated code:

```bash
python -m tools.golden record /tmp/golden.json     # before the change
python -m tools.golden check  /tmp/golden.json     # after: comments and whitespace are ignored
```

## Repository layout

```
spikeduet/
  frontend/snntorch_parser.py   snnTorch nn.Module -> source IR
  ir/
    model.py                    source IR (SNNModel, LinearLayer, LIFLayer)
    verifier.py                 source-IR legality checks
    execution.py                Execution IR (buffers, ops, temporal loop, preheader)
    lowering.py                 source IR -> Execution IR
    execution_verifier.py       Execution IR checks
    execution_analysis.py       def-use queries, temporal invariance
    licm.py, fusion.py          optimization passes
    pass_manager.py             pass infrastructure and named pipelines
    pretty.py, execution_pretty.py   text dumps
  reference/simulator.py        reference executor (ground truth)
  target/platform.py            PlatformSpec (memory, accelerator)
  backend/
    api.py                      compile_model(), backend selection
    base.py                     Backend, GeneratedArtifact
    c_common.py                 model_data.h renderer
    schedule_common.py          op-by-op C emission shared by backends
    portable_c.py               portable C99 backend
    model_header.py             generate_c_model() convenience entry point
    efr32/
      codegen.py                Efr32Backend, CPU and MVP engine generators
      mvp_emitters.py           MVP layer realizations (dense, direct-FP16, range-sparse, windowed)
      cpu_emitters.py           event-driven Cortex-M33 layer realization
      fold.py                   output-affine folding
      profiling.py              DWT stage markers
      cost_model.py             MVP cycle models
      mvp_sparse.py             range-sparse MVP kernel (C text) and hardware constants
      cpu_sparse.py             event-driven Cortex-M33 kernel (C text) and cost model
      placement.py              engine="auto" planner, activity profiles
      reorder.py                hidden-neuron reordering
      async_mvp.py              experimental CPU/MVP overlap
      async_placement.py        experimental overlap cost model
      energy.py                 experimental energy model
  baselines/mlp.py              MLP firmware generator for ANN baselines
runtime/efr32/                  device-side shell (UART, timing, LIF helpers)
tools/hil/                      board protocol, build automation, energy capture
tools/golden.py                 generated-code fingerprinting for refactors
examples/quickstart.py
tests/
```

## Contributing

Run `pytest` before and after a change. If you touch any code generator, also run `tools.golden`. Keep new behaviour opt-in: a new backend option must default to off and leave the generated code unchanged when off.

### Where to change what

| You want to... | Change | Also |
|---|---|---|
| Support a new snnTorch `Leaky` option | `frontend/snntorch_parser.py::_parse_leaky` and a field on `ir/model.py::LIFLayer` | Implement it in `reference/simulator.py`, in the C helpers of `runtime/efr32/snn.h` and in `backend/portable_c.py::_RUNTIME_H` (keep the two identical); emit it in `backend/c_common.py::render_model_data_header`; check it in `ir/verifier.py`; test against live snnTorch in `tests/test_reference.py` |
| Support a new layer type (e.g. Conv) | `frontend/snntorch_parser.py`, `ir/model.py`, `ir/lowering.py` (new op in `ir/execution.py`) | Update `execution_verifier.py`, `execution_analysis.py` (`op_runtime_inputs`/`op_outputs`), and every backend's emitter |
| Add an optimization pass | A new `CompilerPass` subclass in `spikeduet/ir/` | Register a pipeline in `ir/pass_manager.py::PIPELINES`; passes must return a new module and keep the verifier passing |
| Add a backend for a new target | A `Backend` subclass in `spikeduet/backend/` | Register it in `backend/api.py::_BACKENDS`. Reuse `c_common.render_model_data_header` and `schedule_common.emit_schedule`; usually only an `AffineEmitter` is target-specific |
| Change how an affine runs on EFR32 | An emitter in `backend/efr32/mvp_emitters.py` (MVP) or `cpu_emitters.py` (Cortex-M33); the emitted kernels are in `mvp_sparse.py` / `cpu_sparse.py` | Add an `ISCAS_SPARSE_VERIFY` check and a host test that compares against the dense result bitwise |
| Recalibrate cost models for another part | `backend/efr32/cost_model.py`, the constants in `mvp_sparse.py` and `cpu_sparse_affine_cycles` in `cpu_sparse.py` | Re-check `engine="auto"` decisions with `format_placement_report` |
| Describe a new board | `target/platform.py` | Leave unknown limits as `None`, do not guess |
| Change the board protocol | `runtime/efr32/app.c` and `tools/hil/protocol.py` together | `tests/test_hil_protocol.py`, `tests/test_hil_transport_structural.py` |

### Conventions

- **Semantics:** neuron behaviour must match `snntorch==1.0.0` exactly. Add a conformance test against the live module for anything new.
- **Failures:** unsupported input raises an exception naming the offending layer or option; never fall back silently.
- **IR invariants:** IR values are immutable dataclasses. `spikeduet.ir` must not import `spikeduet.target` or any backend, and placement decisions belong to backends.
- **Generated code:** must compile with `-std=c99 -Wall -Wextra -Werror` (portable C) and without warnings on the Arm toolchain.

## Limitations

- **Topology:** the frontend accepts only `[Linear -> Leaky]^N` chains: no convolution, recurrence or skip connections.
- **Batch:** the generated runtimes process one sample at a time.
- **EFR32 coverage:** developed and measured on EFR32MG24 (39 MHz HFXO) and EFM32PG28. Other Series-2 parts with the MVP should work but are untested.
- **Cost models:** calibrated on specific layer shapes. For shapes far from them, treat `engine="auto"` estimates as guidance and measure.
- **MVP numerics:** the MVP computes in FP16. A small fraction of predictions near a decision boundary can differ from the FP32 reference.

## License

MIT, see [LICENSE](LICENSE). `runtime/efr32/main.c` is the Silicon Labs project template and keeps its own Zlib license header.
