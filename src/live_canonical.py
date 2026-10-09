"""Strict primitive-only serialization, independent of module/class identity."""
from collections import deque
from dataclasses import fields, is_dataclass
import hashlib
import json
import math

import numpy as np

HASH_SCHEMA = "live-canonical-json-v1"


def plain(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Canonical state cannot contain non-finite numbers.")
        return float(value)
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.random.Generator):
        return plain(value.bit_generator.state)
    # fields/is_dataclass work on retained instances of pre-hot-reload classes.
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Canonical dictionaries require string keys.")
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [plain(item) for item in value]
    raise TypeError(f"Unsupported canonical state value: {type(value).__name__}.")


def canonical_json(value):
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def live_hash_input(hospital):
    state = {key: value for key, value in hospital.__dict__.items() if key != "_profiles"}
    # Immutable profiles are covered by the existing profile_sha256, avoiding a
    # large duplicate serialization. All other state/counters/history is retained.
    state["_agenda"] = sorted(hospital._agenda)
    state["_free"] = {kind: sorted(ids) for kind, ids in hospital._free.items()}
    state["queue_order"] = [p.patient_id for p in sorted(hospital._waiting.values(), key=hospital.queue_priority)]
    state["waiting_insertion_order"] = list(hospital._waiting)
    state["stress_event_order"] = list(hospital.stress_events)
    state["effective_arrival_rate"] = hospital.effective_arrival_rate
    state["hash_schema"] = HASH_SCHEMA
    state["max_step_minutes"] = hospital.MAX_STEP_MINUTES
    state["rf_threshold"] = .50
    state["policy_history_limit"] = hospital.policy_history.maxlen
    return plain(state)
