# pi0.5 VLA-Fabric

This directory contains the pi0.5 realization used to test whether the
interaction functions discovered with Eagle-style VLA agents transfer to a
structurally different flow-matching VLA. It keeps each arm's observations and
actions local while re-instantiating the three functions as:

- operator-level Raw Common fusion over the multimodal prefix;
- peer Private Prefix K/V consumed by the local action query;
- gated residual peer-action attention recomputed in each layer and flow step.

The released model path is `pi_native_v2_residual_action`. Legacy pi0.5
prototypes, RoboTwin2 adapters, and internal scheduling scripts are intentionally
excluded. The shared N-agent core and focused three- and four-arm task
workflows are included under `scripts/multiarm`.

## External dependencies

Install OpenPI separately and expose its source tree through `OPENPI_ROOT`.
The release does not vendor or modify OpenPI.

```bash
export OPENPI_ROOT=/path/to/openpi
python -m pip install -e "$OPENPI_ROOT"
python -m pip install -e ./pi05
```

Dataset conversion expects the ALOHA RLDS data described in
[`docs/datasets.md`](../docs/datasets.md). Put converted task data under
`pi05/data/converted/<task>` and a pi0.5 base checkpoint under
`pi05/external/pi05_base`, or pass explicit paths to the CLIs.

## Training

The paper protocol uses four processes, one GPU per process, with global batch
12. Stage 1 trains Raw Common for 10k optimizer updates. Stage 2 loads only the
Stage-1 model weights, initializes a fresh optimizer/schedule/RNG, enables all
three interactions, and trains for 40k updates.

```bash
cd pi05
./scripts/train_common_first.sh \
  data/converted/aloha_handover_box \
  external/pi05_base \
  outputs/handover
```

Set `GPUS` and `COORDINATOR_PORT` when the defaults conflict with other jobs.
The Python trainer is dry-run by default; the launcher adds `--execute`.

## Paired evaluation

Development uses condition IDs 0--199. The disjoint paired evaluation uses
IDs 200--399 after checkpoint selection.

```bash
cd pi05
export GPU_ID=1
CUDA_VISIBLE_DEVICES="$GPU_ID" ./scripts/evaluate_paired200.sh \
  outputs/handover/stage2/checkpoint_40000 \
  external/pi05_base \
  data/converted/aloha_handover_box \
  aloha_handover_box \
  outputs/handover/eval_fresh200
```

## Three- and Four-Agent Workflows

The multi-agent extension uses one process and one complete local policy per arm.
See [`docs/multiarm-hpc.md`](../docs/multiarm-hpc.md) for StackCube, Frame
Insertion, Arch Assembly, data auditing, target-HPC readiness, training, and
strict evaluation. GPU IDs are always explicit; no multi-arm launcher has a
default device group. For a new four-GPU allocation, run
`scripts/multiarm/run_four_agent_readiness.sh` before formal training; the
default training scope checks Stage-1-to-Stage-2 transfer, same-stage restore,
finite optimization, and process-specific memory without launching a formal
run.

## Tests

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
JAX_PLATFORMS=cpu \
PYTHONPATH=pi05/src:$OPENPI_ROOT/src \
python -m pytest -q pi05/tests
```
