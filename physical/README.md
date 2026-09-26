# Physical robot data interface

These are the project-owned, hardware-independent recording, validation, and
pi0.5 export utilities. They do not command robot motion or configure devices.
CAN identifiers, camera serials, calibration files, operator consoles, firmware,
and vendor drivers are deliberately not included.

```bash
python -m pip install numpy h5py opencv-python
python physical/scripts/validate_multi_arm_episode.py --help
python physical/scripts/export_multi_arm_pi05.py --help
```

`multi_arm_episode.py` defines timestamped team records with role-local state,
actions, wrist images, and a global camera. `pi05_sampling.py` defines explicit
time-based sampling; `export_multi_arm_pi05.py` validates and transactionally
exports the role datasets with source hashes. Pass source and output paths on
the command line. No sibling project checkout is required.

An exported dataset's manifest may include hardware and source metadata supplied
by the operator. Review those fields separately before releasing data. Do not
upload a raw hardware configuration as part of an anonymous dataset release.

The multi-agent policy computation lives in `pi05/src/pi05_fabric`. Deployment
to a new robot requires its own calibrated action adapter, limits, emergency
stop, and supervised hardware validation.
