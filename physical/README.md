# Physical robot data interface

This package provides hardware-independent recording, validation, and pi0.5
dataset export utilities for multi-arm experiments.

```bash
python -m pip install numpy h5py opencv-python
python physical/scripts/validate_multi_arm_episode.py --help
python physical/scripts/export_multi_arm_pi05.py --help
```

`multi_arm_episode.py` defines timestamped team records with role-local state,
actions, wrist images, and a global camera. `pi05_sampling.py` defines explicit
time-based sampling; `export_multi_arm_pi05.py` validates and transactionally
exports the role datasets with source hashes. Pass source and output paths on
the command line.

The multi-agent policy computation lives in `pi05/src/pi05_fabric`. Deployment
to a new robot requires its own calibrated action adapter, limits, emergency
stop, and supervised hardware validation.
