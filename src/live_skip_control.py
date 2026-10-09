"""Persistent skip debt with bounded stepping and an unthrottled Fast Forward path."""
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

    def fast_forward(self, progress=None):
        """Finish debt without rebuilding the dashboard at observation boundaries.

        Debt is committed before each progress callback, so a Streamlit rerun
        interruption can resume/cancel safely without replaying completed work.
        """
        requested = self.remaining_minutes
        def after_boundary(advanced, total, processed):
            self.remaining_minutes = requested - advanced
            self.chunks_completed += 1
            if progress:
                progress(1 - self.remaining_minutes / self.total_minutes)
        return self.hospital.fast_forward(requested, progress=after_boundary,
            should_continue=lambda: bool(self.remaining_minutes and not self.paused
                and self.hospital.status in ("RUNNING", "PAUSED")))

    def cancel(self):
        self.remaining_minutes = 0.
        self.paused = False
