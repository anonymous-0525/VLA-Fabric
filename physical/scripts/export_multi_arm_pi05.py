#!/usr/bin/env python3
"""Export canonical real-robot episodes to the role-oriented PI0.5 layout."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np

from multi_arm_episode import validate_episode
from pi05_sampling import plan_sampling, validate_sampling


EXPORT_SCHEMA_VERSION = "pi05-multi-arm-v1"
STATE_ADAPTERS = {
    7: "identity_7d",
    9: "right_pad_7d_to_9d_zeros",
}


class ExportInProgressError(RuntimeError):
    """Raised when another exporter owns the destination reservation."""


class _DestinationReservation:
    def __init__(self, destination):
        namespace = _publication_namespace_path(destination)
        self.path = namespace.parent / f".{namespace.name}.export.lock"
        self._fd = None

    def __enter__(self):
        self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(self._fd)
            self._fd = None
            raise ExportInProgressError(
                f"export destination is reserved: {self.path}"
            ) from error
        except BaseException:
            os.close(self._fd)
            self._fd = None
            raise
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


@dataclass(frozen=True)
class ExportReport:
    source: Path
    destination: Path
    manifest: Path
    roles: int
    samples: int
    state_width: int
    source_sha256: str
    training_hz: float
    source_samples: int


@dataclass(frozen=True)
class DynamicRoleSample:
    team_id: str
    trajectory_index: int
    step: int
    role_index: int
    global_rgb: np.ndarray
    wrist_rgb: np.ndarray
    state: np.ndarray
    actions: np.ndarray


@dataclass(frozen=True)
class DynamicTeamSample:
    team_id: str
    trajectory_index: int
    step: int
    roles: tuple[DynamicRoleSample, ...]


def _pad_action_horizon(actions, *, step, horizon):
    chunk = np.asarray(actions[step : step + horizon], dtype=np.float32)
    if not len(chunk):
        raise ValueError("action step is outside the trajectory")
    if len(chunk) < horizon:
        chunk = np.concatenate(
            (chunk, np.repeat(chunk[-1:], horizon - len(chunk), axis=0)),
            axis=0,
        )
    return chunk


class DynamicRoleTaskDataset:
    """Read aligned two-, three-, or four-role PI0.5 team samples."""

    def __init__(self, path, *, action_horizon=50):
        self.path = Path(path).expanduser().resolve()
        self.action_horizon = int(action_horizon)
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        summary = validate_export_pair(self.path)
        self.agent_count = summary["roles"]
        self.sampling = summary["sampling"]
        self.training_hz = None if self.sampling is None else self.sampling["training_hz"]
        self.action_interval_seconds = None if self.training_hz is None else 1.0 / self.training_hz
        with h5py.File(self.path, "r") as handle:
            self.trajectories = tuple(sorted(handle.keys()))
            self.lengths = tuple(
                int(handle[f"{name}/actions/role_0"].shape[0])
                for name in self.trajectories
            )
        self.sample_count = sum(self.lengths)
        self._cumulative_lengths = np.cumsum(self.lengths, dtype=np.int64)
        self._handle = None

    def sample_location(self, rng):
        """Sample one aligned team timestep uniformly over all export steps."""
        flat_index = int(rng.integers(self.sample_count))
        trajectory_index = int(
            np.searchsorted(self._cumulative_lengths, flat_index, side="right")
        )
        previous = (
            0
            if trajectory_index == 0
            else int(self._cumulative_lengths[trajectory_index - 1])
        )
        return trajectory_index, flat_index - previous

    def _file(self):
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self):
        self.close()

    def team_sample(self, *, trajectory_index, step):
        if not 0 <= trajectory_index < len(self.trajectories):
            raise ValueError("trajectory_index is invalid")
        if not 0 <= step < self.lengths[trajectory_index]:
            raise ValueError("step is invalid")
        name = self.trajectories[trajectory_index]
        trajectory = self._file()[name]
        global_rgb = np.asarray(trajectory["global_rgb"][step])
        roles = tuple(
            DynamicRoleSample(
                team_id=name,
                trajectory_index=trajectory_index,
                step=step,
                role_index=role,
                global_rgb=global_rgb,
                wrist_rgb=np.asarray(
                    trajectory[f"wrist_rgb/role_{role}"][step]
                ),
                state=np.asarray(
                    trajectory[f"qpos/role_{role}"][step], dtype=np.float32
                ),
                actions=_pad_action_horizon(
                    trajectory[f"actions/role_{role}"],
                    step=step,
                    horizon=self.action_horizon,
                ),
            )
            for role in range(self.agent_count)
        )
        return DynamicTeamSample(
            team_id=name,
            trajectory_index=trajectory_index,
            step=step,
            roles=roles,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_jpeg(payload, *, stream: str, index: int) -> np.ndarray:
    encoded = np.asarray(payload, dtype=np.uint8).reshape(-1)
    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise ValueError(f"{stream} JPEG frame {index} cannot be decoded")
    rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"{stream} JPEG frame {index} is not RGB uint8")
    if not np.any(rgb):
        raise ValueError(f"{stream} JPEG frame {index} is blank")
    return rgb


def _create_array(group, name, values):
    values = np.asarray(values)
    chunks = (1,) + values.shape[1:]
    return group.create_dataset(
        name,
        data=values,
        chunks=chunks,
        compression="lzf",
    )


def _copy_rgb_stream(source, target_group, name, *, stream: str, source_samples: int, indices):
    if source.shape != (source_samples,):
        raise ValueError(f"{stream} JPEG stream is not team-aligned")
    samples = len(indices)
    first = _decode_jpeg(source[0], stream=stream, index=0)
    dataset = target_group.create_dataset(
        name,
        shape=(samples,) + first.shape,
        dtype=np.uint8,
        chunks=(1,) + first.shape,
        compression="lzf",
    )
    dataset[0] = first
    for index in range(1, samples):
        source_index = int(indices[index])
        frame = _decode_jpeg(source[source_index], stream=stream, index=source_index)
        if frame.shape != first.shape:
            raise ValueError(f"{stream} frame shapes are not consistent")
        dataset[index] = frame
    return dataset


def _adapt_state(values, state_width):
    state = np.asarray(values, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != 7:
        raise ValueError("canonical follower state must have shape [time, 7]")
    if state_width == 7:
        return state
    padding = np.zeros((state.shape[0], 2), dtype=np.float32)
    return np.concatenate((state, padding), axis=1)


def _manifest_path(destination: Path) -> Path:
    return destination.with_suffix(".manifest.json")


def _destination_aliases(destination: Path) -> tuple[Path, Path]:
    destination = Path(destination).expanduser().resolve()
    basename = destination.name[: -len(destination.suffix)]
    return tuple(
        destination.parent / f"{basename}{suffix}"
        for suffix in (".h5", ".hdf5")
    )


def _publication_namespace_path(destination: Path) -> Path:
    destination = Path(destination).expanduser().resolve()
    return _manifest_path(destination)


def _temporary_path(destination: Path) -> Path:
    return destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"


def _transaction_marker_path(destination: Path) -> Path:
    namespace = _publication_namespace_path(destination)
    return namespace.parent / f".{namespace.name}.export-transaction.json"


def _transaction_temp_path(path: Path, publication_id: str) -> Path:
    return path.parent / f".{path.name}.{publication_id}.tmp"


def _fsync_path(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, payload) -> None:
    temporary = _temporary_path(path)
    publication_id = (
        payload.get("publication_id") if isinstance(payload, dict) else None
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if publication_id is not None:
            _unlink_if_owned(temporary, publication_id, manifest=True)


def _artifact_publication_id(path: Path, *, manifest: bool):
    if not path.exists():
        return None
    try:
        if manifest:
            payload = json.loads(path.read_text(encoding="utf-8"))
            value = payload.get("publication_id")
        else:
            with h5py.File(path, "r") as handle:
                value = handle.attrs.get("publication_id")
        return _require_hex(value, 32, "publication_id")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _cleanup_transaction_temps(
    destination: Path, manifest_path: Path, publication_id: str
) -> None:
    _unlink_if_owned(
        _transaction_temp_path(destination, publication_id),
        publication_id,
        manifest=False,
    )
    _unlink_if_owned(
        _transaction_temp_path(manifest_path, publication_id),
        publication_id,
        manifest=True,
    )


def _unlink_if_owned(path: Path, publication_id: str, *, manifest: bool) -> None:
    if _artifact_publication_id(path, manifest=manifest) == publication_id:
        path.unlink(missing_ok=True)


def _recover_interrupted_transaction(
    destination: Path, manifest_path: Path
) -> None:
    marker_path = _transaction_marker_path(destination)
    if not marker_path.exists():
        return
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        publication_id = _require_hex(
            marker["publication_id"], 32, "transaction publication_id"
        )
        source_sha256 = _require_hex(
            marker["source_sha256"], 64, "transaction source hash"
        )
        owner_destination = Path(marker["destination"]).expanduser().resolve()
        owner_manifest = Path(marker["manifest"]).expanduser().resolve()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            "transaction marker is invalid; refusing unsafe recovery"
        ) from error

    aliases = _destination_aliases(destination)
    if (
        owner_destination not in aliases
        or owner_manifest != manifest_path
        or _manifest_path(owner_destination) != manifest_path
    ):
        raise ValueError("transaction marker does not match publication namespace")

    destination_id = _artifact_publication_id(
        owner_destination, manifest=False
    )
    manifest_id = _artifact_publication_id(owner_manifest, manifest=True)
    if destination_id == publication_id and manifest_id == publication_id:
        try:
            summary = validate_export_pair(owner_destination, owner_manifest)
        except ValueError:
            pass
        else:
            if summary["source_sha256"] != source_sha256:
                raise ValueError(
                    "transaction source hash does not match export pair"
                )
            _cleanup_transaction_temps(
                owner_destination, owner_manifest, publication_id
            )
            _unlink_if_owned(marker_path, publication_id, manifest=True)
            _fsync_directory(destination.parent)
            return

    _unlink_if_owned(owner_destination, publication_id, manifest=False)
    _unlink_if_owned(owner_manifest, publication_id, manifest=True)
    _cleanup_transaction_temps(
        owner_destination, owner_manifest, publication_id
    )
    _unlink_if_owned(marker_path, publication_id, manifest=True)
    _fsync_directory(destination.parent)


def _expected_output_roles(role_count):
    return tuple(f"role_{index}" for index in range(role_count))


def _require_nonblank_frames(dataset, label, samples):
    for index in range(samples):
        if not np.any(dataset[index]):
            raise ValueError(f"{label} contains a blank frame at index {index}")


def _require_hex(value, length, label):
    value = str(value)
    if len(value) != length:
        raise ValueError(f"{label} has an invalid length")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} is not hexadecimal") from error
    return value


def _compatible_loaders(role_count):
    loaders = ["export_multi_arm_pi05.DynamicRoleTaskDataset"]
    if role_count == 4:
        loaders.append("pi05_fabric.data.four_arm_tasks.FourArmTaskDataset")
    return loaders


def _validate_export(path: Path):
    """Validate one complete PI0.5 HDF5 artifact."""
    with h5py.File(path, "r") as handle:
        if int(handle.attrs.get("complete", 0)) != 1:
            raise ValueError("export is not marked complete")
        if str(handle.attrs.get("schema_version", "")) != EXPORT_SCHEMA_VERSION:
            raise ValueError("unsupported PI0.5 export schema")
        publication_id = _require_hex(
            handle.attrs.get("publication_id", ""), 32, "publication_id"
        )
        source_sha256 = _require_hex(
            handle.attrs.get("source_sha256", ""), 64, "source hash"
        )
        role_count = int(handle.attrs.get("agent_count", 0))
        if role_count not in (2, 3, 4):
            raise ValueError("PI0.5 export must contain two, three, or four roles")
        samples = int(handle.attrs.get("sample_count", 0))
        state_width = int(handle.attrs.get("state_width", 0))
        if samples < 2 or state_width not in STATE_ADAPTERS:
            raise ValueError("invalid PI0.5 export dimensions")
        state_adapter = str(handle.attrs.get("state_adapter", ""))
        if state_adapter != STATE_ADAPTERS[state_width]:
            raise ValueError("state adapter does not match state width")
        if tuple(handle.keys()) != ("trajectory_000000",):
            raise ValueError("PI0.5 export must contain exactly one trajectory")

        instruction = str(handle.attrs.get("instruction", "")).strip()
        result = str(handle.attrs.get("result", "")).strip()
        task = str(handle.attrs.get("task", "")).strip()
        if not instruction or not result or not task:
            raise ValueError("export instruction, result, and task are required")
        try:
            role_to_hardware = json.loads(
                str(handle.attrs["role_to_hardware_json"])
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid role-to-hardware mapping") from error

        trajectory = handle["trajectory_000000"]
        sampling = validate_sampling(handle, samples)
        for attribute, expected in (
            ("instruction", instruction),
            ("result", result),
            ("task", task),
        ):
            if str(trajectory.attrs.get(attribute, "")) != expected:
                raise ValueError(f"trajectory {attribute} does not match root metadata")
        roles = _expected_output_roles(role_count)
        if set(role_to_hardware) != set(roles):
            raise ValueError("role-to-hardware mapping is not aligned")
        for index, role in enumerate(roles):
            if dict(role_to_hardware[role]).get("source_role") != f"role_{index + 1}":
                raise ValueError("role-to-hardware source ordering is invalid")
        for group_name in ("wrist_rgb", "qpos", "actions"):
            if group_name not in trajectory:
                raise ValueError(f"missing trajectory group: {group_name}")
            if tuple(trajectory[group_name].keys()) != roles:
                raise ValueError(f"{group_name} roles are not aligned")

        global_rgb = trajectory["global_rgb"]
        if (
            global_rgb.shape[0] != samples
            or global_rgb.dtype != np.dtype("uint8")
            or global_rgb.ndim != 4
            or global_rgb.shape[-1] != 3
            or global_rgb.chunks is None
            or global_rgb.compression != "lzf"
        ):
            raise ValueError("global_rgb has an invalid storage contract")
        _require_nonblank_frames(global_rgb, "global_rgb", samples)
        global_dimensions = list(global_rgb.shape)
        wrist_dimensions = {}

        for role in roles:
            wrist = trajectory[f"wrist_rgb/{role}"]
            state = trajectory[f"qpos/{role}"]
            action = trajectory[f"actions/{role}"]
            if (
                wrist.shape[0] != samples
                or wrist.dtype != np.dtype("uint8")
                or wrist.ndim != 4
                or wrist.shape[-1] != 3
                or wrist.chunks is None
                or wrist.compression != "lzf"
            ):
                raise ValueError(f"{role} wrist_rgb has an invalid storage contract")
            _require_nonblank_frames(wrist, f"{role} wrist_rgb", samples)
            wrist_dimensions[role] = list(wrist.shape)
            for dataset, width, label in (
                (state, state_width, "qpos"),
                (action, 7, "actions"),
            ):
                if (
                    dataset.shape != (samples, width)
                    or dataset.dtype != np.dtype("float32")
                    or dataset.chunks is None
                    or dataset.compression != "lzf"
                    or not np.all(np.isfinite(dataset[()]))
                ):
                    raise ValueError(f"{role} {label} has an invalid storage contract")
            if state_width == 9 and not np.array_equal(
                state[:, 7:], np.zeros((samples, 2), dtype=np.float32)
            ):
                raise ValueError(f"{role} state padding is invalid")

    return {
        "roles": role_count,
        "samples": samples,
        "state_width": state_width,
        "state_adapter": state_adapter,
        "publication_id": publication_id,
        "source_sha256": source_sha256,
        "instruction": instruction,
        "result": result,
        "task": task,
        "sampling": sampling,
        "role_to_hardware": role_to_hardware,
        "dimensions": {
            "global_rgb": global_dimensions,
            "wrist_rgb": wrist_dimensions,
            "qpos": [samples, state_width],
            "actions": [samples, 7],
        },
    }


def validate_export_pair(path, manifest_path=None):
    """Validate that PI0.5 HDF5 and sidecar are one matched publication."""
    path = Path(path).expanduser().resolve()
    manifest_path = (
        _manifest_path(path)
        if manifest_path is None
        else Path(manifest_path).expanduser().resolve()
    )
    summary = _validate_export(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("manifest is missing or invalid") from error
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")

    exact_fields = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "publication_id": summary["publication_id"],
        "agent_count": summary["roles"],
        "source_sha256": summary["source_sha256"],
        "state_adapter": summary["state_adapter"],
        "sample_count": summary["samples"],
        "instruction": summary["instruction"],
        "result": summary["result"],
        "task": summary["task"],
        "role_to_hardware": summary["role_to_hardware"],
        "dimensions": summary["dimensions"],
        "compatible_loaders": _compatible_loaders(summary["roles"]),
    }
    if summary["sampling"] is not None or "sampling" in manifest:
        exact_fields["sampling"] = summary["sampling"]
    for field, expected in exact_fields.items():
        if manifest.get(field) != expected:
            label = "source hash" if field == "source_sha256" else field
            raise ValueError(f"manifest {label} does not match HDF5")
    return summary


def _write_export(
    source: Path, target: Path, state_width: int, source_hash: str, publication_id: str,
    training_hz=None,
):
    with h5py.File(source, "r") as canonical, h5py.File(target, "x") as output:
        # Claim the temporary artifact before validation can fail, so transactional
        # cleanup can remove this export's file without touching another writer's.
        output.attrs["publication_id"] = publication_id
        roles = tuple(json.loads(str(canonical.attrs["active_roles_json"])))
        role_count = len(roles)
        if role_count not in (2, 3, 4):
            raise ValueError("PI0.5 export requires exactly two, three, or four active roles")
        expected_roles = tuple(f"role_{index}" for index in range(1, role_count + 1))
        if roles != expected_roles:
            raise ValueError("canonical roles are not contiguous and ordered")
        source_samples = int(canonical.attrs["samples"])
        sampling = plan_sampling(canonical["team/record_monotonic_ns"][()],
                                 canonical.attrs["recording_hz"], training_hz)
        samples = len(sampling.indices)
        hardware = json.loads(str(canonical.attrs["hardware_json"]))
        instruction = str(canonical.attrs["instruction"])
        result = str(canonical.attrs["result"])
        task = str(canonical.attrs["task"])
        adapter = STATE_ADAPTERS[state_width]

        output.attrs.update(
            {
                **sampling.metadata,
                "schema_version": EXPORT_SCHEMA_VERSION,
                "publication_id": publication_id,
                "complete": 0,
                "agent_count": role_count,
                "sample_count": samples,
                "state_width": state_width,
                "state_adapter": adapter,
                "source_sha256": source_hash,
                "instruction": instruction,
                "result": result,
                "task": task,
            }
        )
        trajectory = output.create_group("trajectory_000000")
        timing = trajectory.create_group("timing")
        _create_array(timing, "source_row_index", sampling.indices)
        _create_array(timing, "source_record_monotonic_ns", sampling.source_record_ns)
        _create_array(timing, "timestamp_seconds", sampling.timestamps)
        trajectory.attrs.update(
            {
                "team_id": "trajectory_000000",
                "instruction": instruction,
                "result": result,
                "task": task,
            }
        )
        global_rgb = _copy_rgb_stream(
            canonical["cameras/global/jpeg"],
            trajectory,
            "global_rgb",
            stream="global",
            source_samples=source_samples,
            indices=sampling.indices,
        )
        global_dimensions = list(global_rgb.shape)
        wrists = trajectory.create_group("wrist_rgb")
        qpos = trajectory.create_group("qpos")
        actions = trajectory.create_group("actions")
        role_to_hardware = {}
        wrist_dimensions = {}

        for output_index, source_role in enumerate(roles):
            output_role = f"role_{output_index}"
            wrist = _copy_rgb_stream(
                canonical[f"cameras/wrist_{output_index + 1}/jpeg"],
                wrists,
                output_role,
                stream=f"wrist_{output_index + 1}",
                source_samples=source_samples,
                indices=sampling.indices,
            )
            state = _adapt_state(
                canonical[f"roles/{source_role}/follower/state"][sampling.indices], state_width
            )
            action = np.asarray(
                canonical[f"roles/{source_role}/action/value"][sampling.indices], dtype=np.float32
            )
            if state.shape[0] != samples or action.shape != (samples, 7):
                raise ValueError(f"{source_role} state/action is not team-aligned")
            _create_array(qpos, output_role, state)
            _create_array(actions, output_role, action)
            role_to_hardware[output_role] = {
                "source_role": source_role,
                **dict(hardware[source_role]),
            }
            wrist_dimensions[output_role] = list(wrist.shape)

        output.attrs["role_to_hardware_json"] = json.dumps(
            role_to_hardware, sort_keys=True
        )
        output.flush()
        output.attrs["complete"] = 1
        output.flush()

    return {
        "roles": role_count,
        "samples": samples,
        "sampling": sampling.metadata,
        "instruction": instruction,
        "result": result,
        "task": task,
        "hardware": hardware,
        "role_to_hardware": role_to_hardware,
        "dimensions": {
            "global_rgb": global_dimensions,
            "wrist_rgb": wrist_dimensions,
            "qpos": [samples, state_width],
            "actions": [samples, 7],
        },
    }


def export_episode(source, destination, state_width, *, training_hz=None) -> ExportReport:
    """Convert one complete canonical episode and publish it atomically."""
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    try:
        state_width = int(state_width)
    except (TypeError, ValueError) as error:
        raise ValueError("state_width must be 7 or 9") from error
    if state_width not in STATE_ADAPTERS:
        raise ValueError("state_width must be 7 or 9")
    if source == destination:
        raise ValueError("source and destination must be different files")
    if destination.suffix not in (".h5", ".hdf5"):
        raise ValueError("destination must end in .h5 or .hdf5")

    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_path(destination)
    marker_path = _transaction_marker_path(destination)

    with _DestinationReservation(destination):
        _recover_interrupted_transaction(destination, manifest_path)
        publication_artifacts = (*_destination_aliases(destination), manifest_path)
        if any(path.exists() for path in publication_artifacts):
            raise FileExistsError("destination or manifest already exists")
        if not source.is_file():
            raise FileNotFoundError(f"canonical source does not exist: {source}")

        validate_episode(source)
        source_hash = _sha256(source)
        publication_id = uuid.uuid4().hex
        temporary_hdf5 = _transaction_temp_path(destination, publication_id)
        temporary_manifest = _transaction_temp_path(manifest_path, publication_id)
        transaction = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "publication_id": publication_id,
            "source_sha256": source_hash,
            "destination": str(destination),
            "manifest": str(manifest_path),
            "hdf5_temp": str(temporary_hdf5),
            "manifest_temp": str(temporary_manifest),
            "phase": "converting",
        }
        try:
            _write_json_atomic(marker_path, transaction)
            details = _write_export(
                source,
                temporary_hdf5,
                state_width,
                source_hash,
                publication_id,
                training_hz=training_hz,
            )
            _fsync_path(temporary_hdf5)
            if _sha256(source) != source_hash:
                raise RuntimeError("canonical source changed during export")

            validation = _validate_export(temporary_hdf5)
            expected_validation = {
                "roles": details["roles"],
                "samples": details["samples"],
                "state_width": state_width,
                "state_adapter": STATE_ADAPTERS[state_width],
                "publication_id": publication_id,
                "source_sha256": source_hash,
                "instruction": details["instruction"],
                "result": details["result"],
                "task": details["task"],
                "role_to_hardware": details["role_to_hardware"],
                "dimensions": details["dimensions"],
                "sampling": details["sampling"],
            }
            if validation != expected_validation:
                raise ValueError("export validation summary does not match source")

            command = shlex.join(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    str(source),
                    str(destination),
                    "--state-width",
                    str(state_width),
                    "--training-hz",
                    "original" if training_hz is None else str(float(training_hz)),
                ]
            )
            manifest = {
                "schema_version": EXPORT_SCHEMA_VERSION,
                "publication_id": publication_id,
                "agent_count": details["roles"],
                "source": str(source),
                "destination": str(destination),
                "source_sha256": source_hash,
                "source_references": {
                    "canonical_schema": "scripts/multi_arm_episode.py",
                    "dynamic_loader": (
                        "scripts/export_multi_arm_pi05.py#DynamicRoleTaskDataset"
                    ),
                    "iclr_loader": (
                        "pi05/src/pi05_fabric/data/"
                        "four_arm_tasks.py#FourArmTaskDataset (four roles only)"
                    ),
                },
                "compatible_loaders": _compatible_loaders(details["roles"]),
                "role_to_hardware": details["role_to_hardware"],
                "state_adapter": STATE_ADAPTERS[state_width],
                "dimensions": details["dimensions"],
                "sample_count": details["samples"],
                "sampling": details["sampling"],
                "instruction": details["instruction"],
                "result": details["result"],
                "task": details["task"],
                "conversion_command": command,
            }
            with temporary_manifest.open("x", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())

            prepared = validate_export_pair(temporary_hdf5, temporary_manifest)
            if (
                prepared["publication_id"] != publication_id
                or prepared["source_sha256"] != source_hash
            ):
                raise ValueError("prepared export pair identity does not match transaction")

            transaction["phase"] = "prepared"
            _write_json_atomic(marker_path, transaction)
            os.replace(temporary_manifest, manifest_path)
            _fsync_directory(destination.parent)

            transaction["phase"] = "manifest_published"
            _write_json_atomic(marker_path, transaction)
            os.replace(temporary_hdf5, destination)
            _fsync_directory(destination.parent)

            transaction["phase"] = "artifacts_published"
            _write_json_atomic(marker_path, transaction)
            final = validate_export_pair(destination, manifest_path)
            if (
                final["publication_id"] != publication_id
                or final["source_sha256"] != source_hash
            ):
                raise ValueError("published export pair identity does not match transaction")

            _unlink_if_owned(marker_path, publication_id, manifest=True)
            _fsync_directory(destination.parent)
        except BaseException:
            _unlink_if_owned(
                temporary_hdf5, publication_id, manifest=False
            )
            _unlink_if_owned(
                temporary_manifest, publication_id, manifest=True
            )
            _unlink_if_owned(destination, publication_id, manifest=False)
            _unlink_if_owned(manifest_path, publication_id, manifest=True)
            _unlink_if_owned(marker_path, publication_id, manifest=True)
            _fsync_directory(destination.parent)
            raise

    return ExportReport(
        source=source,
        destination=destination,
        manifest=manifest_path,
        roles=details["roles"],
        samples=details["samples"],
        state_width=state_width,
        source_sha256=source_hash,
        training_hz=details["sampling"]["training_hz"],
        source_samples=details["sampling"]["source_sample_count"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export one canonical multi-arm episode for PI0.5 training."
    )
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--state-width", type=int, choices=(7, 9), default=9)
    parser.add_argument(
        "--training-hz", default="original", metavar="HZ|original",
        help="Training export rate (recommend 10); original keeps every row (default). "
             "Must not exceed source recording Hz. Does not change capture or video.",
    )
    args = parser.parse_args(argv)
    try:
        training_hz = None if args.training_hz == "original" else float(args.training_hz)
    except ValueError:
        parser.error("--training-hz must be a positive number or original")
    report = export_episode(args.source, args.destination, args.state_width, training_hz=training_hz)
    print(
        json.dumps(
            {
                "source": str(report.source),
                "destination": str(report.destination),
                "manifest": str(report.manifest),
                "roles": report.roles,
                "samples": report.samples,
                "state_width": report.state_width,
                "source_sha256": report.source_sha256,
                "training_hz": report.training_hz,
                "source_samples": report.source_samples,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DynamicRoleSample",
    "DynamicRoleTaskDataset",
    "DynamicTeamSample",
    "ExportInProgressError",
    "ExportReport",
    "export_episode",
    "validate_export_pair",
]
