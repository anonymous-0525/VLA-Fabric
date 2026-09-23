# Installation

The Eagle and pi0.5 implementations use different dependency stacks. Install
them in separate Python environments to avoid incompatible PyTorch/JAX and
Transformers requirements.

## Eagle2/OpenVLA environment

Requirements:

- Python 3.10 or 3.11;
- CUDA-capable PyTorch for full training and evaluation;
- an upstream TwinVLA/SingleVLA checkout;
- the ALOHA tabletop simulator used by that checkout.

```bash
python -m venv .venv-eagle
source .venv-eagle/bin/activate
python -m pip install --upgrade pip
python -m pip install -e './eagle[dev]'
```

Place or link external projects beneath `eagle/external/`:

```text
eagle/external/TwinVLA-base/
eagle/external/aloha_rlds/
eagle/external/tabletop_sim/
```

The release never edits these upstream trees.

## pi0.5/OpenPI environment

Requirements:

- Python 3.11;
- a CUDA/JAX installation compatible with the selected accelerator;
- an upstream OpenPI checkout and pi0.5 base checkpoint.

Follow the upstream OpenPI installation instructions first, then install this
wrapper package:

```bash
python -m venv .venv-pi05
source .venv-pi05/bin/activate
python -m pip install --upgrade pip
python -m pip install -e /path/to/openpi
python -m pip install -e ./pi05
export OPENPI_ROOT=/path/to/openpi
```

Expected local-only locations are:

```text
pi05/external/pi05_base/
pi05/data/converted/aloha_handover_box/
pi05/data/converted/aloha_shoes_table/
```

These directories are ignored by Git.

For multi-arm execution, install the external task environment in a separate
compatible environment. StackCube requires RoboFactory and its IPC server
configuration. Frame Insertion and Arch Assembly require the released
`multiarm_sim` task package, robosuite, MuJoCo assets, and an EGL-capable Python
environment. Their paths are passed through the variables documented in
[`multiarm-hpc.md`](multiarm-hpc.md); none is vendored here.

## Verification without accelerators

The protocol tests can run on CPU. Eagle tests do not load the external model.
pi0.5 tests require the OpenPI Python sources because they validate the actual
model boundary.

```bash
PYTHONPATH=eagle/src python -m pytest -q eagle/tests

XLA_FLAGS=--xla_force_host_platform_device_count=4 \
JAX_PLATFORMS=cpu \
PYTHONPATH=pi05/src:$OPENPI_ROOT/src \
python -m pytest -q pi05/tests
```
