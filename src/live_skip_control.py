"""Persistent, cancellable manual skip debt; one bounded chunk per UI update."""
import math

from live_policy_control import MAX_MANUAL_MINUTES


class LiveSkipDriver:
    def __init__(self, hospital):
        self.hospital = hospital
        self.total_minutes = self.remaining_minutes = 0.
        self.paused = False
        self.chunks_completed = 0

    def request(self, minutes):
        if self.hospital.status not in ("RUNNING", "PAUSED"):
            raise ValueError("Start the live twin before requesting a skip.")
        if not math.isfinite(minutes) or not 0 < minutes <= MAX_MANUAL_MINUTES:
            raise ValueError("Manual skip must be between zero and 43200 minutes (30 days).")
        if self.remaining_minutes:
            raise ValueError("Finish or cancel the existing manual skip first.")
        self.total_minutes = self.remaining_minutes = float(minutes)
        self.paused = False
        self.chunks_completed = 0

    def tick(self):
        if self.remaining_minutes and not self.paused and self.hospital.status in ("RUNNING", "PAUSED"):
            delta = min(self.remaining_minutes, self.hospital.MAX_STEP_MINUTES)
            self.hospital.advance_simulation(delta)
            self.remaining_minutes -= delta
            self.chunks_completed += 1
            return delta
        return 0.

    def cancel(self):
        self.remaining_minutes = 0.
        self.paused = False
