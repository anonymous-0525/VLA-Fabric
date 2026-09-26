#!/usr/bin/env python3
"""Atomic, model-independent HDF5 episodes for one to four Piper pairs."""

from __future__ import annotations

import json
import math
import os
import queue
import stat
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import h5py
import numpy as np


SCHEMA_VERSION = 1
FORMAT_NAME = "real_multi_arm_control.canonical_episode"
_STOP = object()


class QueueOverflowError(RuntimeError):
    """Raised when the archive cannot keep up without dropping a row."""


class PublicationCancelled(RuntimeError):
    """Raised after a cancelled publication has been isolated safely."""

    publication_cancelled = True

    def __init__(self, artifact_path):
        self.artifact_path = artifact_path
        super().__init__("episode publication was cancelled and isolated")


@dataclass(frozen=True)
class EpisodeMetadata:
    episode_id: str
    task: str
    instruction: str
    active_roles: tuple[str, ...]
    hardware: Mapping[str, Any]
    software_commit: str
    control_hz: float
    recording_hz: float
    duration_seconds: float
    duration_is_maximum: bool = False
    max_camera_age_seconds: float = 0.25
    max_camera_skew_seconds: float = 0.10


@dataclass(frozen=True)
class EpisodePublication:
    """Published canonical data and its optional derived camera video."""

    hdf5_path: Path
    mp4_path: Path | None
    warning: str | None = None


@dataclass(frozen=True)
class VideoPublication:
    """A published MP4 path and the inode captured before publication."""

    path: Path
    identity: tuple[int, int]
    warning: str | None = None
    status_identity: tuple[int, int] | None = None


@dataclass(frozen=True)
class TeamRow:
    pairs: Mapping[str, Any]
    cameras: Mapping[str, Any]
    record_monotonic_ns: int
    record_wall_time_ns: int


@dataclass(frozen=True)
class _PreparedFrame:
    rgb: np.ndarray
    monotonic_ns: int
    wall_time_ns: int
    hardware_timestamp_ms: float
    frame_number: int


@dataclass(frozen=True)
class _PreparedPair:
    leader: np.ndarray
    leader_monotonic_ns: int
    follower: np.ndarray
    follower_monotonic_ns: int
    action: np.ndarray
    action_monotonic_ns: int


@dataclass(frozen=True)
class _PreparedRow:
    pairs: Mapping[str, _PreparedPair]
    cameras: Mapping[str, _PreparedFrame]
    record_monotonic_ns: int
    record_wall_time_ns: int


def _expected_roles(active_roles):
    roles = tuple(str(role) for role in active_roles)
    if not 1 <= len(roles) <= 4:
        raise ValueError("active_roles must contain one to four roles")
    expected = tuple(f"role_{index}" for index in range(1, len(roles) + 1))
    if roles != expected:
        raise ValueError(f"active roles must be contiguous and ordered: {expected}")
    return roles


def _validate_metadata(metadata):
    if not isinstance(metadata, EpisodeMetadata):
        raise TypeError("metadata must be EpisodeMetadata")
    for name in ("episode_id", "task", "instruction", "software_commit"):
        if not str(getattr(metadata, name)).strip():
            raise ValueError(f"{name} must be non-empty")
    roles = _expected_roles(metadata.active_roles)
    hardware = dict(metadata.hardware)
    if "global" not in hardware or any(role not in hardware for role in roles):
        raise ValueError("hardware mapping must include global and every active role")
    if not str(dict(hardware["global"]).get("serial", "")).strip():
        raise ValueError("global hardware mapping must include a camera serial")
    required_role_hardware = (
        "leader_can_serial",
        "follower_can_serial",
        "wrist_camera_serial",
    )
    for role in roles:
        mapping = dict(hardware[role])
        if any(
            not str(mapping.get(name, "")).strip()
            for name in required_role_hardware
        ):
            raise ValueError(f"{role} hardware mapping is incomplete")
    for name in ("control_hz", "recording_hz"):
        value = float(getattr(metadata, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(float(metadata.duration_seconds)) or metadata.duration_seconds < 0:
        raise ValueError("duration_seconds must be finite and nonnegative")
    if not isinstance(metadata.duration_is_maximum, bool):
        raise ValueError("duration_is_maximum must be boolean")
    for name in ("max_camera_age_seconds", "max_camera_skew_seconds"):
        value = float(getattr(metadata, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    json.dumps(hardware, sort_keys=True)
    return roles


def encode_rgb_jpeg(image, quality=90):
    """Encode one HWC RGB uint8 image as a one-dimensional uint8 JPEG."""
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an HWC uint8 RGB array")
    quality = int(quality)
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be in [1, 100]")
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok or encoded is None or encoded.size == 0:
        raise RuntimeError("JPEG encoding failed")
    return np.asarray(encoded, dtype=np.uint8).reshape(-1)


def _decode_jpeg_frame(payload, role, index):
    encoded = np.asarray(payload, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError(f"camera {role} JPEG frame {index} cannot be decoded")
    return frame


def _validate_mp4(path, *, expected_frames, expected_size):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError("published MP4 cannot be opened")
        reported_frames = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if reported_frames != expected_frames:
            raise RuntimeError(
                "published MP4 frame count mismatch: "
                f"expected {expected_frames}, got {reported_frames}"
            )
        reported_size = (
            round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if reported_size != expected_size:
            raise RuntimeError(
                "published MP4 frame size mismatch: "
                f"expected {expected_size}, got {reported_size}"
            )
        ok, first = capture.read()
        if not ok or first is None or first.size == 0:
            raise RuntimeError("published MP4 first frame cannot be decoded")
        if expected_frames > 1:
            capture.set(cv2.CAP_PROP_POS_FRAMES, expected_frames - 1)
            ok, last = capture.read()
            if not ok or last is None or last.size == 0:
                raise RuntimeError("published MP4 last frame cannot be decoded")
    finally:
        capture.release()


def _move_no_replace(source, destination, *, purpose):
    """Move a same-filesystem file without replacing an existing target."""
    try:
        os.link(source, destination, follow_symlinks=False)
    except FileExistsError as error:
        raise RuntimeError(
            f"{purpose} refused to overwrite {destination}; "
            f"source is preserved at {source}"
        ) from error
    except OSError as error:
        raise RuntimeError(
            f"{purpose} could not create a no-clobber link at {destination}; "
            f"source is preserved at {source}: {error}"
        ) from error
    try:
        os.unlink(source)
    except OSError as error:
        raise RuntimeError(
            f"{purpose} linked {destination}, but could not remove {source}; "
            "both links are preserved"
        ) from error


def _file_identity(path):
    result = os.lstat(path)
    if not stat.S_ISREG(result.st_mode):
        raise RuntimeError(f"path is not a regular file: {path}")
    return result.st_dev, result.st_ino


def _remove_empty_directory(path):
    try:
        Path(path).rmdir()
    except OSError:
        pass


def _claim_owned_path(path, expected_identity, *, purpose):
    path = Path(path)
    claim_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{path.name}.claim-",
            dir=path.parent,
        )
    )
    claimed = claim_directory / path.name
    try:
        try:
            os.replace(path, claimed)
        except FileNotFoundError:
            return None, claim_directory
        claimed_stat = os.lstat(claimed)
        actual = (claimed_stat.st_dev, claimed_stat.st_ino)
        is_regular = stat.S_ISREG(claimed_stat.st_mode)
        if not is_regular or actual != expected_identity:
            _move_no_replace(
                claimed,
                path,
                purpose=f"foreign {purpose} restoration",
            )
            if not is_regular:
                raise RuntimeError(
                    f"{purpose} is not a regular file; refusing to use it"
                )
            raise RuntimeError(
                f"{purpose} identity changed; refusing to use a foreign file"
            )
        return claimed, claim_directory
    except BaseException:
        _remove_empty_directory(claim_directory)
        raise


def export_camera_mosaic_mp4(
    hdf5_path,
    video_path,
    *,
    camera_roles=None,
    cancel_event=None,
    progress=None,
):
    """Derive a validated equal-cell multi-pair review video from HDF5 rows."""
    from episode_mosaic import compose_episode_mosaic, mosaic_roles

    source = Path(hdf5_path).expanduser()
    destination = Path(video_path).expanduser()
    if destination.suffix.lower() != ".mp4":
        raise ValueError("video path must end in .mp4")
    if destination.exists():
        raise FileExistsError(f"video already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.partial-",
        suffix=".mp4",
        dir=destination.parent,
    )
    identity_descriptor = None
    try:
        temporary_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(temporary_stat.st_mode):
            raise RuntimeError("temporary MP4 is not a regular file")
        temporary_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
        # Keep the inode alive even if the public temporary path is replaced.
        identity_descriptor = os.dup(file_descriptor)
    finally:
        os.close(file_descriptor)
    temporary = Path(temporary_name)
    claim_directory = None
    temporary_is_claimed = False
    writer = None
    try:
        with h5py.File(source, "r") as episode:
            if int(episode.attrs.get("complete", 0)) != 1:
                raise RuntimeError("source HDF5 episode is not complete")
            roles = tuple(episode.get("cameras", {}).keys()) if camera_roles is None else tuple(camera_roles)
            hardware = json.loads(episode.attrs.get("hardware_json", "{}"))
            mosaic_roles(roles, hardware)
            datasets = []
            for role in roles:
                path = f"cameras/{role}/jpeg"
                if path not in episode:
                    raise RuntimeError(f"source HDF5 is missing camera {role}")
                datasets.append(episode[path])
            frame_count = len(datasets[0])
            if frame_count <= 0 or any(
                len(dataset) != frame_count for dataset in datasets
            ):
                raise RuntimeError("camera frame counts are empty or misaligned")
            fps = float(episode.attrs.get("recording_hz", 0.0))
            if not math.isfinite(fps) or fps <= 0:
                raise RuntimeError("source HDF5 recording_hz is invalid")

            first_frames = [
                _decode_jpeg_frame(dataset[0], role, 0)
                for role, dataset in zip(roles, datasets)
            ]
            first_mosaic = compose_episode_mosaic(dict(zip(roles, first_frames)), hardware)
            frame_size = (first_mosaic.shape[1], first_mosaic.shape[0])
            writer = cv2.VideoWriter(
                str(temporary),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                frame_size,
            )
            if not writer.isOpened():
                raise RuntimeError("OpenCV could not open the mp4v video encoder")
            if progress is not None:
                progress(0, frame_count)

            for index in range(frame_count):
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("MP4 export was cancelled")
                if index == 0:
                    mosaic = first_mosaic
                else:
                    frames = [
                        _decode_jpeg_frame(dataset[index], role, index)
                        for role, dataset in zip(roles, datasets)
                    ]
                    mosaic = compose_episode_mosaic(dict(zip(roles, frames)), hardware)
                if (mosaic.shape[1], mosaic.shape[0]) != frame_size:
                    raise RuntimeError("camera mosaic shape changed during MP4 export")
                writer.write(mosaic)
                if progress is not None:
                    progress(index + 1, frame_count)
        writer.release()
        writer = None
        claimed, claim_directory = _claim_owned_path(
            temporary,
            temporary_identity,
            purpose="temporary MP4",
        )
        if claimed is None:
            raise RuntimeError("temporary MP4 disappeared before validation")
        temporary = claimed
        temporary_is_claimed = True
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError("MP4 encoder produced an empty file")
        _validate_mp4(
            temporary,
            expected_frames=frame_count,
            expected_size=frame_size,
        )
        _move_no_replace(
            temporary,
            destination,
            purpose="MP4 publication",
        )
        return VideoPublication(destination, temporary_identity)
    except BaseException as error:
        if writer is not None:
            writer.release()
        if temporary_is_claimed:
            temporary.unlink(missing_ok=True)
        else:
            cleanup_directory = None
            try:
                owned, cleanup_directory = _claim_owned_path(
                    temporary,
                    temporary_identity,
                    purpose="temporary MP4 cleanup",
                )
                if owned is not None:
                    owned.unlink()
            except BaseException as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note(
                        "temporary MP4 cleanup was refused: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            finally:
                if cleanup_directory is not None:
                    _remove_empty_directory(cleanup_directory)
        raise
    finally:
        if claim_directory is not None:
            _remove_empty_directory(claim_directory)
        if identity_descriptor is not None:
            os.close(identity_descriptor)


def _finite_vector(values, width, name):
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (width,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape ({width},) with finite values")
    return array.copy()


def _positive_timestamp(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _frame_field(frame, primary, fallback=None):
    if hasattr(frame, primary):
        return getattr(frame, primary)
    if fallback and hasattr(frame, fallback):
        return getattr(frame, fallback)
    raise ValueError(f"camera frame is missing {primary}")


class EpisodeWriter:
    """Append complete team rows to a bounded asynchronous HDF5 writer."""

    def __init__(
        self,
        path,
        metadata,
        *,
        queue_rows=64,
        jpeg_quality=90,
        worker_gate=None,
        worker_join_timeout=30.0,
        mosaic_video_path=None,
        mosaic_camera_roles=("global", "wrist_1"),
    ):
        self.path = Path(path).expanduser()
        if self.path.suffix != ".hdf5":
            raise ValueError("episode path must end in .hdf5")
        if self.path.exists():
            raise FileExistsError(f"episode already exists: {self.path}")
        self.partial_path = Path(f"{self.path}.partial")
        if self.partial_path.exists():
            raise FileExistsError(f"partial episode already exists: {self.partial_path}")
        self.metadata = metadata
        self.roles = _validate_metadata(metadata)
        self.camera_roles = ("global",) + tuple(
            f"wrist_{index}" for index in range(1, len(self.roles) + 1)
        )
        queue_rows = int(queue_rows)
        if queue_rows <= 0:
            raise ValueError("queue_rows must be positive")
        self.jpeg_quality = int(jpeg_quality)
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        self._queue = queue.Queue(maxsize=queue_rows)
        self._worker_gate = worker_gate
        self._worker_join_timeout = float(worker_join_timeout)
        self._lock = threading.RLock()
        self._state = "open"
        self._fatal_error = None
        self._accepted_rows = 0
        self._written_rows = 0
        self._last_timestamps = {}
        self._worker_closed = False
        self._stop_requested = False
        self._deferred_isolation = None
        self._isolation_claimed = False
        self._isolated_path = None
        self._isolation_error = None
        self._published_identity = None
        self.mosaic_video_path = (
            None
            if mosaic_video_path is None
            else Path(mosaic_video_path).expanduser()
        )
        self.mosaic_camera_roles = tuple(mosaic_camera_roles)
        self._published_video_identity = None
        self._published_video_status_identity = None
        if self.mosaic_video_path is not None:
            if self.mosaic_video_path.suffix.lower() != ".mp4":
                raise ValueError("mosaic video path must end in .mp4")
            if self.mosaic_video_path.exists():
                raise FileExistsError(
                    f"mosaic video already exists: {self.mosaic_video_path}"
                )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._create_partial()
        self._thread = threading.Thread(
            target=self._worker,
            name=f"episode-writer-{self.path.stem}",
            daemon=True,
        )
        try:
            self._thread.start()
        except BaseException:
            self.partial_path.unlink(missing_ok=True)
            raise

    def _create_partial(self):
        with h5py.File(self.partial_path, "x") as episode:
            episode.attrs.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "format": FORMAT_NAME,
                    "complete": 0,
                    "result": "recording",
                    "abort_reason": "",
                    "episode_id": self.metadata.episode_id,
                    "task": self.metadata.task,
                    "instruction": self.metadata.instruction,
                    "active_roles_json": json.dumps(list(self.roles)),
                    "hardware_json": json.dumps(
                        dict(self.metadata.hardware), sort_keys=True
                    ),
                    "software_commit": self.metadata.software_commit,
                    "control_hz": float(self.metadata.control_hz),
                    "recording_hz": float(self.metadata.recording_hz),
                    "duration_seconds": float(self.metadata.duration_seconds),
                    "duration_is_maximum": bool(
                        self.metadata.duration_is_maximum
                    ),
                    "max_camera_age_seconds": float(
                        self.metadata.max_camera_age_seconds
                    ),
                    "max_camera_skew_seconds": float(
                        self.metadata.max_camera_skew_seconds
                    ),
                    "samples": 0,
                }
            )
            team = episode.create_group("team")
            self._create_numeric(team, "record_monotonic_ns", (), np.int64)
            self._create_numeric(team, "record_wall_time_ns", (), np.int64)
            self._create_numeric(team, "max_camera_age_ns", (), np.int64)
            self._create_numeric(team, "camera_skew_ns", (), np.int64)

            role_root = episode.create_group("roles")
            for role in self.roles:
                role_group = role_root.create_group(role)
                for component in ("leader", "follower"):
                    group = role_group.create_group(component)
                    self._create_numeric(group, "state", (7,), np.float32)
                    self._create_numeric(group, "monotonic_ns", (), np.int64)
                action = role_group.create_group("action")
                self._create_numeric(action, "value", (7,), np.float32)
                self._create_numeric(action, "monotonic_ns", (), np.int64)

            camera_root = episode.create_group("cameras")
            for camera_role in self.camera_roles:
                group = camera_root.create_group(camera_role)
                jpeg = group.create_dataset(
                    "jpeg",
                    shape=(0,),
                    maxshape=(None,),
                    chunks=(1,),
                    dtype=h5py.vlen_dtype(np.dtype("uint8")),
                )
                jpeg.attrs["codec"] = "jpeg"
                jpeg.attrs["quality"] = self.jpeg_quality
                self._create_numeric(group, "monotonic_ns", (), np.int64)
                self._create_numeric(group, "wall_time_ns", (), np.int64)
                self._create_numeric(group, "hardware_timestamp_ms", (), np.float64)
                self._create_numeric(group, "frame_number", (), np.int64)
                self._create_numeric(group, "frame_age_ns", (), np.int64)
            episode.flush()

    @staticmethod
    def _create_numeric(group, name, tail_shape, dtype):
        tail_shape = tuple(tail_shape)
        chunks = (64,) + tail_shape
        return group.create_dataset(
            name,
            shape=(0,) + tail_shape,
            maxshape=(None,) + tail_shape,
            chunks=chunks,
            dtype=dtype,
            compression="lzf",
        )

    def _prepare_row(self, row):
        if not isinstance(row, TeamRow):
            raise TypeError("row must be TeamRow")
        pair_keys = tuple(f"pair_{index}" for index in range(1, len(self.roles) + 1))
        if set(row.pairs) == set(self.roles):
            pairs_by_role = {
                role: (row.pairs[role], role) for role in self.roles
            }
        elif set(row.pairs) == set(pair_keys):
            pairs_by_role = {
                role: (row.pairs[pair_key], pair_key)
                for role, pair_key in zip(self.roles, pair_keys)
            }
        else:
            raise ValueError("row pairs must align exactly with active roles")
        if set(row.cameras) != set(self.camera_roles):
            raise ValueError("row cameras must align exactly with active roles")
        record_ns = _positive_timestamp(row.record_monotonic_ns, "record_monotonic_ns")
        wall_ns = _positive_timestamp(row.record_wall_time_ns, "record_wall_time_ns")

        pairs = {}
        source_timestamps = {"record_monotonic_ns": record_ns}
        for role in self.roles:
            snapshot, source_pair_id = pairs_by_role[role]
            if str(getattr(snapshot, "pair_id", source_pair_id)) != source_pair_id:
                raise ValueError(f"{role} pair_id does not align with active roles")
            leader_joint = _finite_vector(snapshot.leader_joint, 6, f"{role} leader_joint")
            follower_joint = _finite_vector(
                snapshot.follower_joint, 6, f"{role} follower_joint"
            )
            leader = np.concatenate(
                (leader_joint, _finite_vector([snapshot.leader_gripper], 1, "leader gripper"))
            )
            follower = np.concatenate(
                (
                    follower_joint,
                    _finite_vector([snapshot.follower_gripper], 1, "follower gripper"),
                )
            )
            action = _finite_vector(snapshot.action, 7, f"{role} action")
            leader_ns = _positive_timestamp(
                snapshot.leader_monotonic_ns, f"{role} leader timestamp"
            )
            follower_ns = _positive_timestamp(
                snapshot.follower_monotonic_ns, f"{role} follower timestamp"
            )
            action_ns = _positive_timestamp(
                snapshot.action_monotonic_ns, f"{role} action timestamp"
            )
            if max(leader_ns, follower_ns, action_ns) > record_ns:
                raise ValueError("state and action timestamps cannot be newer than the row")
            source_timestamps.update(
                {
                    f"{role}/leader": leader_ns,
                    f"{role}/follower": follower_ns,
                    f"{role}/action": action_ns,
                }
            )
            pairs[role] = _PreparedPair(
                leader=leader,
                leader_monotonic_ns=leader_ns,
                follower=follower,
                follower_monotonic_ns=follower_ns,
                action=action,
                action_monotonic_ns=action_ns,
            )

        cameras = {}
        camera_times = []
        for camera_role in self.camera_roles:
            frame = row.cameras[camera_role]
            frame_role = str(_frame_field(frame, "role"))
            if frame_role != camera_role:
                raise ValueError(f"camera {camera_role} role is misaligned: {frame_role}")
            rgb = np.asarray(_frame_field(frame, "rgb"))
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(f"{camera_role} must provide an HWC uint8 RGB image")
            if not np.any(rgb):
                raise ValueError(f"{camera_role} RGB image is blank")
            frame_ns = _positive_timestamp(
                _frame_field(frame, "host_monotonic_ns", "monotonic_ns"),
                f"{camera_role} timestamp",
            )
            frame_wall_ns = _positive_timestamp(
                _frame_field(frame, "host_wall_time_ns", "wall_time_ns"),
                f"{camera_role} wall timestamp",
            )
            hardware_ms = float(_frame_field(frame, "hardware_timestamp_ms"))
            frame_number = int(_frame_field(frame, "frame_number"))
            if not math.isfinite(hardware_ms) or hardware_ms <= 0 or frame_number < 0:
                raise ValueError(f"{camera_role} frame metadata is invalid")
            age_ns = record_ns - frame_ns
            if age_ns < 0:
                raise ValueError(f"{camera_role} camera timestamp is newer than the row")
            if age_ns > round(self.metadata.max_camera_age_seconds * 1e9):
                raise ValueError(f"{camera_role} camera frame is stale")
            source_timestamps[f"camera/{camera_role}"] = frame_ns
            camera_times.append(frame_ns)
            cameras[camera_role] = _PreparedFrame(
                rgb=rgb.copy(),
                monotonic_ns=frame_ns,
                wall_time_ns=frame_wall_ns,
                hardware_timestamp_ms=hardware_ms,
                frame_number=frame_number,
            )
        if max(camera_times) - min(camera_times) > round(
            self.metadata.max_camera_skew_seconds * 1e9
        ):
            raise ValueError("camera timestamp skew exceeds configured maximum")

        with self._lock:
            for source, timestamp in source_timestamps.items():
                previous = self._last_timestamps.get(source)
                if previous is not None and timestamp <= previous:
                    raise ValueError(f"{source} timestamp must be strictly increasing")
            self._last_timestamps.update(source_timestamps)
        return _PreparedRow(pairs, cameras, record_ns, wall_ns)

    def append(self, row):
        with self._lock:
            if self._state != "open":
                detail = f": {self._fatal_error}" if self._fatal_error else ""
                raise RuntimeError(f"episode writer is not open{detail}")
            prepared = self._prepare_row(row)
            try:
                self._queue.put_nowait(prepared)
            except queue.Full as error:
                overflow = QueueOverflowError(
                    "writer queue overflow; no row was dropped silently"
                )
                self._fatal_error = overflow
                self._state = "failed"
                raise overflow from error
            self._accepted_rows += 1

    def _worker(self):
        try:
            if self._worker_gate is not None:
                self._worker_gate.wait()
            with h5py.File(self.partial_path, "r+") as episode:
                while True:
                    item = self._queue.get()
                    try:
                        if item is _STOP:
                            break
                        self._write_row(episode, item)
                        self._written_rows += 1
                    finally:
                        self._queue.task_done()
                    with self._lock:
                        stop_when_drained = (
                            self._stop_requested and self._queue.empty()
                        )
                    if stop_when_drained:
                        break
                episode.attrs["samples"] = self._written_rows
                episode.flush()
        except BaseException as error:
            with self._lock:
                if self._fatal_error is None:
                    self._fatal_error = error
                if self._state == "open":
                    self._state = "failed"
        finally:
            self._mark_worker_closed()

    def _mark_worker_closed(self):
        with self._lock:
            self._worker_closed = True
            deferred = self._deferred_isolation
        if deferred is None:
            return
        try:
            self._isolate_once(*deferred)
        except BaseException as error:
            with self._lock:
                self._isolation_error = error

    def _write_row(self, episode, row):
        index = self._written_rows
        numeric_values = {
            "team/record_monotonic_ns": row.record_monotonic_ns,
            "team/record_wall_time_ns": row.record_wall_time_ns,
        }
        camera_times = [frame.monotonic_ns for frame in row.cameras.values()]
        numeric_values["team/max_camera_age_ns"] = max(
            row.record_monotonic_ns - timestamp for timestamp in camera_times
        )
        numeric_values["team/camera_skew_ns"] = max(camera_times) - min(camera_times)
        for role, pair in row.pairs.items():
            numeric_values.update(
                {
                    f"roles/{role}/leader/state": pair.leader,
                    f"roles/{role}/leader/monotonic_ns": pair.leader_monotonic_ns,
                    f"roles/{role}/follower/state": pair.follower,
                    f"roles/{role}/follower/monotonic_ns": pair.follower_monotonic_ns,
                    f"roles/{role}/action/value": pair.action,
                    f"roles/{role}/action/monotonic_ns": pair.action_monotonic_ns,
                }
            )
        for path, value in numeric_values.items():
            dataset = episode[path]
            dataset.resize(index + 1, axis=0)
            dataset[index] = value

        for camera_role, frame in row.cameras.items():
            group = episode[f"cameras/{camera_role}"]
            jpeg = group["jpeg"]
            shape = tuple(int(value) for value in frame.rgb.shape)
            previous_shape = tuple(jpeg.attrs.get("original_shape", ()))
            if previous_shape and previous_shape != shape:
                raise ValueError(f"{camera_role} RGB shape changed during the episode")
            if not previous_shape:
                jpeg.attrs["original_shape"] = shape
            encoded = encode_rgb_jpeg(frame.rgb, self.jpeg_quality)
            jpeg.resize(index + 1, axis=0)
            jpeg[index] = encoded
            values = {
                "monotonic_ns": frame.monotonic_ns,
                "wall_time_ns": frame.wall_time_ns,
                "hardware_timestamp_ms": frame.hardware_timestamp_ms,
                "frame_number": frame.frame_number,
                "frame_age_ns": row.record_monotonic_ns - frame.monotonic_ns,
            }
            for name, value in values.items():
                dataset = group[name]
                dataset.resize(index + 1, axis=0)
                dataset[index] = value

    def _finish_worker(self):
        with self._lock:
            self._stop_requested = True
        if self._thread.is_alive():
            if self._worker_gate is not None:
                self._worker_gate.set()
            deadline = time.monotonic() + self._worker_join_timeout
            while self._thread.is_alive():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    self._queue.put(_STOP, timeout=min(0.05, remaining))
                    break
                except queue.Full:
                    continue
            self._thread.join(self._worker_join_timeout)
        if self._thread.is_alive():
            raise RuntimeError("episode writer thread did not stop")

    def _claim_finalization(self):
        with self._lock:
            if self._state == "closing":
                raise RuntimeError("episode finalization is already in progress")
            if self._state not in ("open", "failed"):
                raise RuntimeError("episode writer is already finalized")
            self._state = "closing"

    def _fail_finalization(self, result, error):
        reason = str(error)
        with self._lock:
            if self._fatal_error is None:
                self._fatal_error = error
            self._state = "failed_final"
            if not self._worker_closed:
                self._deferred_isolation = (result, reason)
                return None
        try:
            return self._isolate_once(result, reason)
        except BaseException as isolate_error:
            if hasattr(error, "add_note"):
                error.add_note(f"failed to isolate partial episode: {isolate_error}")
            return None

    def _failed_path(self):
        directory = self.path.parent / "failed_artifacts"
        directory.mkdir(parents=True, exist_ok=True)
        candidate = directory / self.partial_path.name
        counter = 1
        while candidate.exists():
            candidate = directory / f"{self.partial_path.name}.{counter}"
            counter += 1
        return candidate

    def _isolate_once(self, result, reason):
        with self._lock:
            if self._isolation_claimed:
                return self._isolated_path
            self._isolation_claimed = True
        try:
            failed = self._isolate(result, reason)
        except BaseException as error:
            with self._lock:
                self._isolation_error = error
            raise
        with self._lock:
            self._isolated_path = failed
        return failed

    def _isolate(self, result, reason):
        if not self.partial_path.exists():
            return None
        try:
            with h5py.File(self.partial_path, "r+") as episode:
                episode.attrs["complete"] = 0
                episode.attrs["result"] = str(result)
                episode.attrs["abort_reason"] = str(reason)
                episode.attrs["samples"] = self._written_rows
                episode.flush()
        except OSError:
            pass
        failed = self._failed_path()
        os.replace(self.partial_path, failed)
        return failed

    def _cancel_publication_if_requested(self, cancel_event, *, published=False):
        if cancel_event is None or not cancel_event.is_set():
            return
        try:
            if published:
                self._restore_published_partial()
            failed = self._isolate_once(
                "aborted", "episode publication cancelled before finalization"
            )
        except BaseException:
            with self._lock:
                self._state = "failed_final"
            raise
        with self._lock:
            self._state = "aborted"
        raise PublicationCancelled(failed)

    @staticmethod
    def _file_identity(path):
        return _file_identity(path)

    @staticmethod
    def _move_no_replace(source, destination, *, purpose):
        return _move_no_replace(source, destination, purpose=purpose)

    @staticmethod
    def _remove_empty_directory(path):
        _remove_empty_directory(path)

    def _restore_published_partial(self):
        """Claim by rename, then verify identity before isolation.

        The claim directory is private and on the destination filesystem, so
        ``os.replace`` atomically captures whichever inode owns the final name.
        Restoration uses hard-link creation as a portable no-clobber primitive;
        filesystems without same-filesystem hard links fail closed and preserve
        the claimed file. This protects path replacement, not in-place mutation
        by another process with access to the same inode or private directory.
        """
        with self._lock:
            expected = self._published_identity
        if expected is None:
            raise RuntimeError("published artifact identity is unavailable")

        claim_directory = Path(
            tempfile.mkdtemp(
                prefix=f".{self.path.name}.rollback-",
                dir=self.path.parent,
            )
        )
        claimed = claim_directory / self.path.name
        try:
            try:
                os.replace(self.path, claimed)
            except FileNotFoundError as error:
                raise RuntimeError("published artifact is missing") from error

            actual = self._file_identity(claimed)
            if actual != expected:
                try:
                    self._move_no_replace(
                        claimed,
                        self.path,
                        purpose="foreign artifact restoration",
                    )
                except BaseException as restore_error:
                    raise RuntimeError(
                        "published artifact identity changed; refusing isolation; "
                        f"foreign file is preserved at {claimed}; "
                        f"safe restoration failed: {restore_error}"
                    ) from restore_error
                raise RuntimeError(
                    "published artifact identity changed; refusing isolation; "
                    "foreign file was restored without overwriting the final path"
                )

            self._move_no_replace(
                claimed,
                self.partial_path,
                purpose="published artifact rollback",
            )
            with self._lock:
                self._published_identity = None
        finally:
            self._remove_empty_directory(claim_directory)

    def rollback_publication(self, reason):
        """Idempotently isolate this writer's already-published artifact."""
        with self._lock:
            if self._state == "aborted" and self._isolated_path is not None:
                return self._isolated_path
            if self._state != "published":
                raise RuntimeError("episode writer has no published artifact to roll back")
            self._state = "rolling_back"
        try:
            self._remove_published_video()
            self._restore_published_partial()
            failed = self._isolate_once("aborted", str(reason))
        except BaseException:
            with self._lock:
                self._state = "failed_final"
            raise
        with self._lock:
            self._state = "aborted"
        return failed

    def _remove_published_video(self):
        if self.mosaic_video_path is None:
            return
        for path, attribute in (
            (self.mosaic_video_path, "_published_video_identity"),
            (self.mosaic_video_path.with_suffix(".json"), "_published_video_status_identity"),
        ):
            expected = getattr(self, attribute)
            if expected is None:
                continue
            claimed, claim_directory = _claim_owned_path(path, expected, purpose="published video artifact")
            try:
                if claimed is not None:
                    claimed.unlink()
                setattr(self, attribute, None)
            finally:
                self._remove_empty_directory(claim_directory)

    def publish(self, result, *, cancel_event=None, progress=None):
        self._claim_finalization()
        try:
            self._finish_worker()
        except BaseException as error:
            self._fail_finalization("failed", error)
            raise
        with self._lock:
            fatal = self._fatal_error
        if fatal is not None:
            self._fail_finalization("failed", fatal)
            raise fatal
        if self._written_rows != self._accepted_rows:
            error = RuntimeError("writer row count mismatch")
            self._fail_finalization("failed", error)
            raise error
        self._cancel_publication_if_requested(cancel_event)
        try:
            with h5py.File(self.partial_path, "r+") as episode:
                episode.attrs["complete"] = 1
                episode.attrs["result"] = str(result)
                episode.attrs["abort_reason"] = ""
                episode.attrs["samples"] = self._written_rows
                episode.flush()
            validate_episode(self.partial_path)
            self._cancel_publication_if_requested(cancel_event)
            published_identity = self._file_identity(self.partial_path)
            with self._lock:
                self._published_identity = published_identity
            os.replace(self.partial_path, self.path)
            self._cancel_publication_if_requested(cancel_event, published=True)
        except PublicationCancelled:
            raise
        except BaseException as error:
            self._fail_finalization("invalid", error)
            raise
        video_path = None
        warning = None
        if self.mosaic_video_path is not None:
            try:
                from episode_artifacts import export_episode_video
                video = export_episode_video(
                    self.path,
                    self.mosaic_video_path,
                    camera_roles=self.mosaic_camera_roles,
                    cancel_event=cancel_event,
                    progress=progress,
                )
                video_path = video.path
                self._published_video_identity = video.identity
                self._published_video_status_identity = video.status_identity
                warning = video.warning
            except Exception as error:
                self._published_video_status_identity = getattr(error, "video_status_identity", None)
                warning = f"MP4 export failed: {type(error).__name__}: {error}"
            if cancel_event is not None and cancel_event.is_set():
                self._remove_published_video()
                self._cancel_publication_if_requested(cancel_event, published=True)
        with self._lock:
            self._state = "published"
        if self.mosaic_video_path is None:
            return self.path
        return EpisodePublication(self.path, video_path, warning)

    def abort(self, reason):
        self._claim_finalization()
        try:
            self._finish_worker()
        except BaseException as error:
            self._fail_finalization("failed", error)
            raise
        with self._lock:
            fatal = self._fatal_error
        if fatal is not None:
            self._fail_finalization("failed", fatal)
            raise fatal
        try:
            failed = self._isolate_once("aborted", str(reason))
        except BaseException:
            with self._lock:
                self._state = "failed_final"
            raise
        with self._lock:
            self._state = "aborted"
        return failed


def _strictly_increasing(values):
    values = np.asarray(values)
    return values.ndim == 1 and len(values) > 1 and bool(np.all(np.diff(values) > 0))


def _require_positive(values, label):
    values = np.asarray(values)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"{label} must be positive")


def _require_numeric_attr(episode, name, *, positive=False, nonnegative=False):
    try:
        value = float(episode.attrs[name])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _require_dataset(episode, path, expected_shape, samples, expected_dtype):
    if path not in episode:
        raise ValueError(f"missing dataset: {path}")
    dataset = episode[path]
    if dataset.shape != (samples,) + tuple(expected_shape):
        raise ValueError(
            f"{path} shape {dataset.shape} does not match "
            f"{(samples,) + tuple(expected_shape)}"
        )
    expected_dtype = np.dtype(expected_dtype)
    if dataset.dtype != expected_dtype:
        raise ValueError(
            f"{path} dtype {dataset.dtype} does not match {expected_dtype}"
        )
    if dataset.compression != "lzf":
        raise ValueError(f"{path} must use LZF compression")
    return dataset


def _parse_json_attr(episode, name):
    try:
        return json.loads(str(episode.attrs[name]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {name}") from error


def _decode_checked_frames(dataset, expected_shape, stream):
    count = len(dataset)
    for index in sorted({0, count // 2, count - 1}):
        payload = np.asarray(dataset[index], dtype=np.uint8)
        decoded_bgr = cv2.imdecode(payload, cv2.IMREAD_COLOR)
        if decoded_bgr is None:
            raise ValueError(f"{stream} JPEG frame {index} cannot be decoded")
        decoded = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
        if tuple(decoded.shape) != expected_shape:
            raise ValueError(f"{stream} JPEG frame {index} has wrong shape")
        if not np.any(decoded):
            raise ValueError(f"{stream} JPEG frame {index} is blank")


def validate_episode(path):
    """Validate a complete canonical episode and return a JSON-safe summary."""
    path = Path(path).expanduser()
    with h5py.File(path, "r") as episode:
        if int(episode.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("unsupported schema version")
        if str(episode.attrs.get("format", "")) != FORMAT_NAME:
            raise ValueError("unsupported episode format")
        if int(episode.attrs.get("complete", 0)) != 1:
            raise ValueError("episode is not complete")

        for name in ("episode_id", "task", "instruction", "software_commit"):
            if not str(episode.attrs.get(name, "")).strip():
                raise ValueError(f"missing metadata: {name}")
        result = str(episode.attrs.get("result", "")).strip()
        if not result or result in {"recording", "aborted", "invalid"}:
            raise ValueError("result is invalid for a complete published episode")
        if "abort_reason" not in episode.attrs:
            raise ValueError("missing metadata: abort_reason")
        if str(episode.attrs["abort_reason"]).strip():
            raise ValueError("abort_reason must be empty for a complete episode")

        control_hz = _require_numeric_attr(episode, "control_hz", positive=True)
        recording_hz = _require_numeric_attr(
            episode, "recording_hz", positive=True
        )
        requested_duration = _require_numeric_attr(
            episode, "duration_seconds", nonnegative=True
        )
        raw_duration_is_maximum = episode.attrs.get(
            "duration_is_maximum", False
        )
        if not isinstance(raw_duration_is_maximum, (bool, np.bool_)):
            raise ValueError("duration_is_maximum must be boolean")
        duration_is_maximum = bool(raw_duration_is_maximum)
        max_camera_age_seconds = _require_numeric_attr(
            episode, "max_camera_age_seconds", positive=True
        )
        max_camera_skew_seconds = _require_numeric_attr(
            episode, "max_camera_skew_seconds", positive=True
        )

        roles = _expected_roles(_parse_json_attr(episode, "active_roles_json"))
        hardware = _parse_json_attr(episode, "hardware_json")
        if "global" not in hardware or any(role not in hardware for role in roles):
            raise ValueError("hardware role mapping is incomplete")
        if not str(dict(hardware["global"]).get("serial", "")).strip():
            raise ValueError("global hardware role mapping is incomplete")
        required_hardware = (
            "leader_can_serial",
            "follower_can_serial",
            "wrist_camera_serial",
        )
        for role in roles:
            mapping = dict(hardware[role])
            if any(
                not str(mapping.get(name, "")).strip()
                for name in required_hardware
            ):
                raise ValueError(f"{role} hardware role mapping is incomplete")
        if "roles" not in episode or set(episode["roles"].keys()) != set(roles):
            raise ValueError("stored role groups do not align with active roles")
        expected_cameras = {"global"} | {
            f"wrist_{index}" for index in range(1, len(roles) + 1)
        }
        if (
            "cameras" not in episode
            or set(episode["cameras"].keys()) != expected_cameras
        ):
            raise ValueError("stored camera groups do not align with active roles")

        samples = int(episode.attrs.get("samples", -1))
        if samples < 2:
            raise ValueError("episode must contain at least two samples")
        record_ns = _require_dataset(
            episode, "team/record_monotonic_ns", (), samples, np.int64
        )[()]
        record_wall_ns = _require_dataset(
            episode, "team/record_wall_time_ns", (), samples, np.int64
        )[()]
        max_age_rows = _require_dataset(
            episode, "team/max_camera_age_ns", (), samples, np.int64
        )[()]
        skew_rows = _require_dataset(
            episode, "team/camera_skew_ns", (), samples, np.int64
        )[()]
        _require_positive(record_ns, "team record timestamps")
        _require_positive(record_wall_ns, "team wall timestamps")
        if not _strictly_increasing(record_ns):
            raise ValueError("team record timestamps must be strictly increasing")
        if not _strictly_increasing(record_wall_ns):
            raise ValueError("team wall timestamps must be strictly increasing")
        if np.any(max_age_rows < 0) or np.any(skew_rows < 0):
            raise ValueError("camera diagnostics cannot be negative")

        for role in roles:
            for component, value_name in (
                ("leader", "state"),
                ("follower", "state"),
                ("action", "value"),
            ):
                base = f"roles/{role}/{component}"
                values = _require_dataset(
                    episode,
                    f"{base}/{value_name}",
                    (7,),
                    samples,
                    np.float32,
                )[()]
                timestamps = _require_dataset(
                    episode,
                    f"{base}/monotonic_ns",
                    (),
                    samples,
                    np.int64,
                )[()]
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"{base} contains non-finite values")
                _require_positive(timestamps, f"{base} timestamps")
                if not _strictly_increasing(timestamps):
                    raise ValueError(f"{base} timestamps must be strictly increasing")
                if np.any(timestamps > record_ns):
                    raise ValueError(f"{base} timestamps are newer than team rows")

        camera_timestamps = []
        for stream in sorted(expected_cameras):
            base = f"cameras/{stream}"
            jpeg = episode[f"{base}/jpeg"]
            if (
                jpeg.shape != (samples,)
                or h5py.check_dtype(vlen=jpeg.dtype) != np.dtype("uint8")
            ):
                raise ValueError(f"{base}/jpeg must be variable-length uint8")
            if str(jpeg.attrs.get("codec", "")) != "jpeg":
                raise ValueError(f"{base}/jpeg codec must be jpeg")
            shape = tuple(int(value) for value in jpeg.attrs.get("original_shape", ()))
            if len(shape) != 3 or shape[2] != 3 or min(shape) <= 0:
                raise ValueError(f"{base}/jpeg original shape is invalid")
            _decode_checked_frames(jpeg, shape, stream)
            timestamps = _require_dataset(
                episode, f"{base}/monotonic_ns", (), samples, np.int64
            )[()]
            wall_times = _require_dataset(
                episode, f"{base}/wall_time_ns", (), samples, np.int64
            )[()]
            hardware_times = _require_dataset(
                episode,
                f"{base}/hardware_timestamp_ms",
                (),
                samples,
                np.float64,
            )[()]
            frame_numbers = _require_dataset(
                episode, f"{base}/frame_number", (), samples, np.int64
            )[()]
            frame_ages = _require_dataset(
                episode, f"{base}/frame_age_ns", (), samples, np.int64
            )[()]
            _require_positive(timestamps, f"{base} timestamps")
            _require_positive(wall_times, f"{base} wall timestamps")
            _require_positive(hardware_times, f"{base} hardware timestamps")
            if np.any(frame_numbers < 0):
                raise ValueError(f"{base} frame numbers cannot be negative")
            for values, label in (
                (timestamps, "timestamps"),
                (wall_times, "wall timestamps"),
                (hardware_times, "hardware timestamps"),
                (frame_numbers, "frame numbers"),
            ):
                if not _strictly_increasing(values):
                    raise ValueError(f"{base} {label} must be strictly increasing")
            expected_age = record_ns - timestamps
            if np.any(expected_age < 0) or not np.array_equal(
                frame_ages, expected_age
            ):
                raise ValueError(f"{base} frame age diagnostics are invalid")
            camera_timestamps.append(timestamps)

        camera_matrix = np.stack(camera_timestamps, axis=0)
        computed_skew = camera_matrix.max(axis=0) - camera_matrix.min(axis=0)
        computed_age = (record_ns[None, :] - camera_matrix).max(axis=0)
        if not np.array_equal(computed_skew, skew_rows) or not np.array_equal(
            computed_age, max_age_rows
        ):
            raise ValueError("camera timing diagnostics do not match source timestamps")
        max_age_ns = round(max_camera_age_seconds * 1e9)
        max_skew_ns = round(max_camera_skew_seconds * 1e9)
        if int(computed_age.max()) > max_age_ns:
            raise ValueError("camera frame age exceeds configured maximum")
        if int(computed_skew.max()) > max_skew_ns:
            raise ValueError("camera skew exceeds configured maximum")

        span_seconds = float(record_ns[-1] - record_ns[0]) / 1e9
        if span_seconds <= 0:
            raise ValueError("episode duration must be positive")
        if (
            not duration_is_maximum
            and requested_duration > 0
            and span_seconds < requested_duration * 0.90
        ):
            raise ValueError(
                f"episode duration {span_seconds:.3f}s does not cover "
                f"requested duration {requested_duration:.3f}s"
            )
        if duration_is_maximum and requested_duration > 0:
            maximum_tolerance = (1.0 / recording_hz) + 0.01
            if span_seconds > requested_duration + maximum_tolerance:
                raise ValueError(
                    f"episode duration {span_seconds:.3f}s exceeds maximum duration "
                    f"{requested_duration:.3f}s"
                )
        effective_hz = (samples - 1) / span_seconds
        if not recording_hz * 0.75 <= effective_hz <= recording_hz * 1.25:
            raise ValueError(
                f"effective sample rate {effective_hz:.2f}Hz is invalid for "
                f"{recording_hz:.2f}Hz"
            )
        return {
            "path": str(path),
            "schema_version": SCHEMA_VERSION,
            "episode_id": str(episode.attrs["episode_id"]),
            "task": str(episode.attrs["task"]),
            "result": result,
            "samples": samples,
            "active_roles": list(roles),
            "camera_streams": len(expected_cameras),
            "duration_seconds": span_seconds,
            "maximum_duration_seconds": (
                requested_duration if duration_is_maximum else None
            ),
            "effective_recording_hz": effective_hz,
            "control_hz": control_hz,
            "max_camera_age_ms": float(computed_age.max()) / 1e6,
            "max_camera_skew_ms": float(computed_skew.max()) / 1e6,
        }


__all__ = [
    "EpisodeMetadata",
    "EpisodeWriter",
    "PublicationCancelled",
    "QueueOverflowError",
    "TeamRow",
    "encode_rgb_jpeg",
    "validate_episode",
]
