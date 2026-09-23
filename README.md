# VLA-Fabric: ICLR 2027 Code Release

Project page: https://anonymous-0525.github.io/VLA-Fabric/

This repository contains the code used for the ICLR 2027 VLA-Fabric study:
discovering how complete VLA policies learn to coordinate, re-instantiating
the resulting interaction functions in a structurally different VLA backbone,
and extending the same one-arm-one-agent organization to three- and four-arm
tasks.

## Released implementations

| Directory | Backbone | Purpose |
|---|---|---|
| [`eagle/`](eagle/) | Eagle2/OpenVLA-style SingleVLA | Controlled architecture and training study over 32 task-model combinations |
| [`pi05/`](pi05/) | pi0.5/OpenPI | Architecture-aware transfer and single-node 2/3/4-agent workflows |

Both implementations preserve one-arm-one-agent ownership: each policy receives
its local observation and produces its local action. Peer information crosses
only the interaction boundaries defined by the experiment configuration.

The release includes the three-agent StackCube workflow and four-agent Frame
Insertion and Arch Assembly adapters. It excludes checkpoints, datasets,
rollouts, internal cluster queues, and the communication-efficiency/NSPR
implementation studied separately.

## Quick start

1. Follow [`docs/installation.md`](docs/installation.md) for the two independent
   environments and upstream dependencies.
2. Prepare the ALOHA tasks as described in
   [`docs/datasets.md`](docs/datasets.md).
3. Use the package-level READMEs for training and paired evaluation.
4. For three- and four-agent execution, follow [`docs/multiarm-hpc.md`](docs/multiarm-hpc.md).
5. Consult [`docs/reproducibility.md`](docs/reproducibility.md) for the exact
   architecture matrix, stage protocol, and evaluation split.

Run the lightweight release checks with:

```bash
python -m pytest -q tests
python scripts/audit_release.py
```

Backbone-specific tests are documented in [`eagle/README.md`](eagle/README.md)
and [`pi05/README.md`](pi05/README.md).

## Repository status

This is a research release accompanying a work in progress. The project page
shows selected, completed evaluations; it is not a model zoo. Model weights,
datasets, and complete rollout archives are not bundled. Paths in the
checked-in YAML files are portable placeholders rather than references to
the authors' machines.

## License

Project-owned code is released under the MIT License in
[`LICENSE`](LICENSE). TwinVLA, OpenPI, ALOHA
datasets, simulators, and pretrained checkpoints remain subject to their own
licenses and must be obtained from their original sources.
