"""Common-row, timestamp-based sampling for derived training artifacts only."""

from dataclasses import dataclass

import numpy as np


# Accommodate nanosecond rounding at an otherwise exact endpoint, not an extra frame.
TIME_EPSILON_SECONDS = 1e-6
SAMPLING_FIELDS = (
    "sampling_version", "resampling_method", "source_recording_hz", "training_hz",
    "source_sample_count", "source_record_start_ns", "max_time_error_seconds",
)


@dataclass(frozen=True)
class SamplingPlan:
    indices: np.ndarray
    timestamps: np.ndarray
    source_record_ns: np.ndarray
    metadata: dict


def plan_sampling(record_ns, source_hz, training_hz=None):
    """Select whole observations; never interpolate images/actions or repeat rows.

    Explicit rates use a regular grid starting at the first source row, selecting
    the nearest source row (ties choose the earlier one). None preserves all rows
    and their actual timing. All arrays are shared by every role and camera.
    """
    record_ns = np.asarray(record_ns)
    source_hz = float(source_hz)
    hz = source_hz if training_hz is None else float(training_hz)
    if not np.isfinite(source_hz) or source_hz <= 0:
        raise ValueError("source recording frequency must be finite and positive")
    if not np.isfinite(hz) or hz <= 0 or hz > source_hz:
        raise ValueError("training_hz must be positive, finite, and <= source recording_hz")
    if (record_ns.ndim != 1 or record_ns.dtype != np.dtype("int64")
            or len(record_ns) < 2 or np.any(record_ns <= 0)
            or np.any(np.diff(record_ns) <= 0)):
        raise ValueError("sampling requires increasing int64 source timestamps")
    # Subtract integers before conversion: large monotonic epochs retain ns precision.
    seconds = (record_ns - record_ns[0]).astype(np.float64) / 1e9
    if training_hz is None:
        indices = np.arange(len(record_ns), dtype=np.int64)
        timestamps = seconds
        method = "original"
    else:
        count = int(np.floor((seconds[-1] + TIME_EPSILON_SECONDS) * hz)) + 1
        if count < 2:
            raise ValueError("sampling would produce fewer than two rows; choose a higher Hz")
        if count > len(record_ns):
            raise ValueError("sampling requires more observations than the source provides")
        timestamps = np.arange(count, dtype=np.float64) / hz
        right = np.searchsorted(seconds, timestamps).clip(0, len(seconds) - 1)
        left = (right - 1).clip(0)
        indices = np.where(
            np.abs(seconds[right] - timestamps) < np.abs(seconds[left] - timestamps) - 1e-12,
            right, left,
        ).astype(np.int64)
        if np.any(np.diff(indices) <= 0):
            raise ValueError("sampling would repeat source rows across a gap; use a lower Hz or original")
        method = "nearest-common-row"
    max_error = float(np.max(np.abs(seconds[indices] - timestamps)))
    if max_error > 0.5 / hz + TIME_EPSILON_SECONDS:
        raise ValueError("source gap exceeds half a training sampling interval")
    return SamplingPlan(indices, timestamps, record_ns[indices], {
        "sampling_version": 1,
        "resampling_method": method,
        "source_recording_hz": source_hz,
        "training_hz": hz,
        "source_sample_count": len(record_ns),
        "source_record_start_ns": int(record_ns[0]),
        "max_time_error_seconds": max_error,
    })


def validate_sampling(handle, samples):
    """Validate the additive v1 timing extension; old exports remain readable."""
    trajectory = handle["trajectory_000000"]
    present = [field in handle.attrs for field in SAMPLING_FIELDS]
    if not any(present) and "timing" not in trajectory:
        return None  # Historical exports did not record a time base; do not guess it.
    if not all(present) or "timing" not in trajectory:
        raise ValueError("incomplete sampling metadata")
    meta = {field: handle.attrs[field] for field in SAMPLING_FIELDS}
    for name in ("sampling_version", "source_sample_count", "source_record_start_ns"):
        if not isinstance(meta[name], (int, np.integer)):
            raise ValueError(f"sampling {name} must be an integer")
        meta[name] = int(meta[name])
    for name in ("source_recording_hz", "training_hz", "max_time_error_seconds"):
        meta[name] = float(meta[name])
        if not np.isfinite(meta[name]):
            raise ValueError(f"sampling {name} must be finite")
    method = str(meta["resampling_method"])
    meta["resampling_method"] = method
    hz = meta["training_hz"]
    if (meta["sampling_version"] != 1 or method not in ("original", "nearest-common-row")
            or not 0 < hz <= meta["source_recording_hz"]
            or meta["source_sample_count"] < samples or meta["source_record_start_ns"] <= 0):
        raise ValueError("invalid sampling metadata")
    timing = trajectory["timing"]
    arrays = {}
    for name, dtype in (("source_row_index", "int64"), ("source_record_monotonic_ns", "int64"),
                        ("timestamp_seconds", "float64")):
        if name not in timing:
            raise ValueError(f"missing sampling timing/{name}")
        dataset = timing[name]
        if dataset.shape != (samples,) or dataset.dtype != np.dtype(dtype):
            raise ValueError(f"invalid sampling timing/{name}")
        arrays[name] = dataset[()]
    indices = arrays["source_row_index"]
    record_ns = arrays["source_record_monotonic_ns"]
    timestamps = arrays["timestamp_seconds"]
    if (indices[0] != 0 or indices[-1] >= meta["source_sample_count"]
            or np.any(np.diff(indices) <= 0)
            or record_ns[0] != meta["source_record_start_ns"] or np.any(np.diff(record_ns) <= 0)
            or not np.all(np.isfinite(timestamps)) or timestamps[0] != 0
            or np.any(np.diff(timestamps) <= 0)):
        raise ValueError("sampling rows/timestamps must be aligned and strictly increasing")
    source_seconds = (record_ns - record_ns[0]).astype(np.float64) / 1e9
    if method == "original":
        if (meta["source_sample_count"] != samples or hz != meta["source_recording_hz"]
                or not np.array_equal(indices, np.arange(samples))
                or not np.array_equal(timestamps, source_seconds)):
            raise ValueError("original sampling must preserve all source rows and timing")
    elif not np.allclose(timestamps, np.arange(samples) / hz, rtol=0, atol=1e-12):
        raise ValueError("sampling timeline does not match training_hz")
    max_error = float(np.max(np.abs(source_seconds - timestamps)))
    if (abs(max_error - meta["max_time_error_seconds"]) > 1e-12
            or max_error > 0.5 / hz + TIME_EPSILON_SECONDS):
        raise ValueError("sampling time error does not match metadata or exceeds half an interval")
    return meta
