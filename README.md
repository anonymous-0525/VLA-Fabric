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
| [`simulation/`](simulation/) | MuJoCo/robosuite | Three-arm frame, four-arm frame, and arch task environments |
| [`physical/`](physical/) | Hardware-independent | Timestamped recordings, validation, and role-dataset export |

Both implementations preserve one-arm-one-agent ownership: each policy receives
its local observation and produces its local action. Peer information crosses
only the interaction boundaries defined by the experiment configuration.

The release includes StackCube, three-/four-arm frame tasks, Arch Assembly,
RoboTwin bimanual task adapters, independently trained controls, and inference
path removals. It excludes checkpoints, raw datasets, complete rollout archives,
internal cluster queues, and the communication-efficiency/NSPR
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

## Interactive project page

The GitHub Pages source is in `docs/`; the Three.js scene and synchronized
camera player are in `site-src/`. Local builds do not require the ML environments:

```bash
npm ci
npm run build
npm run preview
# http://localhost:4173
```

The page includes the animated four-arm illustration, four real-robot recordings
with synchronized privacy-filtered wrist views, and global-view simulation clips.
See [`docs/media.md`](docs/media.md) for source distinctions, processing, and QA.

## Repository status

This is a core-code research release. The project page
shows selected, completed evaluations; it is not a model zoo. Model weights,
datasets, and complete rollout archives are not bundled. Paths in the
checked-in YAML files are portable placeholders rather than references to
the authors' machines. Hardware drivers and device registration are intentionally
separate from the portable data and policy interfaces.

## License

Project-owned code is released under the MIT License in
[`LICENSE`](LICENSE). TwinVLA, OpenPI, ALOHA
datasets, simulators, and pretrained checkpoints remain subject to their own
licenses and must be obtained from their original sources.
