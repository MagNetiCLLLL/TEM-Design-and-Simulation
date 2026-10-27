"""Explicit work progress, independent of physics, Qt and elapsed time.

Local counts describe one solver step. Overall progress counts only declared,
completed workflow stages; local completion cannot complete a workflow stage.
Legacy three-argument callbacks remain supported through a string-compatible
label carrying immutable metadata. Plain labels never imply overall progress.
"""
from dataclasses import dataclass, replace
from operator import index


def _count(value):
    return max(0, index(value))


@dataclass(frozen=True)
class LocalProgress:
    completed: int
    total: int | None

    def __post_init__(self):
        if self.completed < 0 or (self.total is not None and (
                self.total <= 0 or self.completed > self.total)):
            raise ValueError("Local work requires nonnegative counts and a valid known total")


class LocalProgressLabel(str):
    """Local counter metadata for text-only transport callback interfaces."""
    def __new__(cls, text, completed, total):
        value = super().__new__(cls, text)
        count = _count(total) if total is not None else 0
        object.__setattr__(value, "local_progress", LocalProgress(
            min(_count(completed), count) if count else _count(completed), count or None))
        return value

    def __setattr__(self, name, value):
        raise AttributeError("Progress labels are immutable")


@dataclass(frozen=True)
class StageProgress:
    completed: int
    total: int
    name: str
    local_completed: int | None = None
    local_total: int | None = None

    def __post_init__(self):
        if self.total <= 0 or not 0 <= self.completed <= self.total:
            raise ValueError("Stage progress requires a positive total and bounded completion")
        if self.local_completed is None and self.local_total is not None:
            raise ValueError("A known total requires a local work count")
        if self.local_completed is not None and self.local_completed < 0:
            raise ValueError("Local work counts must be nonnegative")
        if self.local_total is not None and (
                self.local_total <= 0 or not 0 <= self.local_completed <= self.local_total):
            raise ValueError("Local work counts must be bounded by a positive total")


class StageProgressLabel(str):
    """A normal callback label with explicit stage and current-step counts."""
    def __new__(cls, text, progress):
        if not isinstance(progress, StageProgress):
            raise TypeError("StageProgressLabel requires StageProgress metadata")
        value = super().__new__(cls, text)
        object.__setattr__(value, "progress", progress)
        return value

    def __setattr__(self, name, value):
        raise AttributeError("Progress labels are immutable")


@dataclass(frozen=True)
class CalculationProgress:
    operation_id: str
    operation: str
    step: str
    completed: int | None = None
    total: int | None = None
    stages_completed: int | None = None
    stages_total: int | None = None
    stage_name: str = ""
    status: str = "running"
    parent_id: str | None = None
    pair_error: str | None = None

    def __post_init__(self):
        if self.status not in {"running", "succeeded", "failed", "cancelled"}:
            raise ValueError("Unknown calculation progress status")
        for position, (done, total) in enumerate(((self.completed, self.total),
                            (self.stages_completed, self.stages_total))):
            if (done is None and total is not None) or (position == 1 and (done is None) != (total is None)):
                raise ValueError("Known progress requires both completion and total")
            if done is not None and done < 0:
                raise ValueError("Completed work must be nonnegative")
            if total is not None and (total <= 0 or not 0 <= done <= total):
                raise ValueError("Progress counts must be bounded by a positive total")

    @property
    def active(self):
        return self.status == "running"

    def terminal(self, status, message=None):
        if status == "running":
            raise ValueError("A terminal progress status is required")
        # Only accepted publication can mark all stages complete. Failures and
        # cancellations retain actual completion instead of filling the bar.
        return replace(self, status=status, step=message or status.capitalize(),
            stages_completed=(self.stages_total if status == "succeeded"
                              else self.stages_completed))


def callback_progress(previous, completed, total, label, *, reserved_stages=0):
    """Preserve unknown totals and never infer whole-job work from local counts."""
    details = getattr(label, "progress", None)
    if isinstance(details, StageProgress):
        return replace(previous, step=str(label),
            completed=details.local_completed, total=details.local_total,
            stages_completed=details.completed,
            stages_total=details.total + reserved_stages, stage_name=details.name)
    local = getattr(label, "local_progress", None)
    if isinstance(local, LocalProgress):
        return replace(previous, step=str(label), completed=local.completed, total=local.total)
    count = _count(total) if total is not None else 0
    return replace(previous, step=str(label),
        completed=min(_count(completed), count) if count else _count(completed),
        total=count or None)


def bounded_bar_counts(completed, total, *, maximum=1_000_000):
    """Map large counters onto Qt without overflowing its signed integers."""
    if total is None:
        return 0, 0
    if total <= maximum:
        return completed, total
    # Integer arithmetic avoids precision loss and premature rounded 100%.
    return completed * maximum // total, maximum
