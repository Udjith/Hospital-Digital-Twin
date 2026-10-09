"""Pure display helpers. Engine values and serialized artifacts stay in minutes."""
import math
import re


def format_duration(minutes):
    """Compact duration; nearest second below an hour, nearest minute above.

    Half units round upward and carry into the next unit. Missing/invalid values
    display an em dash. Zero minute components are omitted only for day values.
    """
    if minutes is None or not math.isfinite(minutes) or minutes < 0:
        return "—"
    if minutes < 60:
        seconds = math.floor(minutes * 60 + .5)
        if seconds < 60:
            return f"{seconds}s"
        if seconds < 3600:
            whole_minutes, seconds = divmod(seconds, 60)
            return f"{whole_minutes}m {seconds}s"
    whole_minutes = math.floor(minutes + .5)
    days, remainder = divmod(whole_minutes, 1440)
    hours, remainder = divmod(remainder, 60)
    if days:
        return f"{days}d {hours}h" + (f" {remainder}m" if remainder else "")
    return f"{hours}h {remainder}m"


def format_timestamp(minutes):
    """Session-relative timestamp: time zero is Day 1 00:00:00."""
    if minutes is None or not math.isfinite(minutes) or minutes < 0:
        return "—"
    seconds = math.floor(minutes * 60 + .5)
    day, seconds = divmod(seconds, 86400)
    hour, seconds = divmod(seconds, 3600)
    minute, second = divmod(seconds, 60)
    return f"Day {day + 1} {hour:02d}:{minute:02d}:{second:02d}"


def format_time_text(text):
    """Humanize minute durations in displayed prose, preserving stored text."""
    return re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:minutes|min)\b",
                  lambda match: format_duration(float(match[1])), text)
