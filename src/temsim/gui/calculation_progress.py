"""A truthful two-line status display for independently owned calculations."""
from collections import OrderedDict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

from temsim.calculation_progress import CalculationProgress, bounded_bar_counts


class CalculationProgressWidget(QWidget):
    """Show the current step separately from completed workflow stages.

    Multiple concurrent captured requests retain their own progress. Completing
    one request cannot hide another or replace its counters with 100 percent.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("calculationProgressDetails")
        self.current = QProgressBar(self)
        self.current.setObjectName("currentCalculationStep")
        self.overall = QProgressBar(self)
        self.overall.setObjectName("overallCalculationProgress")
        self.current_label = QLabel(self)
        self.overall_label = QLabel(self)
        for label in (self.current_label, self.overall_label):
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse
                                          | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addWidget(self.current_label)
        layout.addWidget(self.current)
        layout.addWidget(self.overall_label)
        layout.addWidget(self.overall)
        for bar in (self.current, self.overall):
            bar.setMaximumHeight(6)
            bar.setTextVisible(False)
        self.setMinimumWidth(380)
        self.setMaximumWidth(700)
        self._updates = OrderedDict()
        self.displayed = None
        self.hide()
        self.setToolTip("Current-step counts describe only the named work. Overall progress counts "
                        "completed workflow stages, not elapsed time or an estimate of time remaining. "
                        "Unknown work remains indeterminate. Completion requires accepted output.")

    def set_progress(self, update):
        if not isinstance(update, CalculationProgress):
            raise TypeError("Expected CalculationProgress")
        previous = self._updates.get(update.operation_id)
        if previous is not None and not previous.active and update.active:
            return  # Late queued signals cannot resurrect a cancelled job.
        self._updates[update.operation_id] = update
        self._updates.move_to_end(update.operation_id)
        active = [value for value in self._updates.values() if value.active]
        # Pair owners already aggregate their children's work. Keep ordinary
        # standalone jobs visible, but do not count a pair and child twice.
        owners = {value.operation_id for value in active}
        visible = [value for value in active if value.parent_id not in owners]
        selected = visible[-1] if visible else update
        self.displayed = selected
        prefix = selected.operation + (f" ({len(visible)} jobs active)" if len(visible) > 1 else "")
        self._set_bar(self.current, selected.completed, selected.total, selected.active)
        local = (f"{selected.completed}/{selected.total}" if selected.total is not None
                 else f"{selected.completed} completed; total unknown" if selected.completed is not None
                 else "progress unknown" if selected.active else selected.status.capitalize())
        self.current.setFormat(f"{prefix} | Current: {selected.step} | {local}")
        self.current_label.setText(self.current.format())
        self._set_bar(self.overall, selected.stages_completed, selected.stages_total, selected.active)
        overall = (f"{selected.stages_completed}/{selected.stages_total} stages complete"
                   if selected.stages_total is not None else
                   "Complete" if selected.status == "succeeded" else "progress unknown")
        if selected.status == "succeeded" and selected.stages_total is None:
            # Acceptance proves whole-job completion even when the solver
            # could not declare its number of stages in advance.
            self.overall.setRange(0, 1)
            self.overall.setValue(1)
        status = "" if selected.active else f" | {selected.status.capitalize()}"
        self.overall.setFormat(f"Overall: {overall}{status}")
        self.overall_label.setText(self.overall.format())
        self.show()
        # Bound terminal UI history while preserving all active requests.
        terminal = [key for key, value in self._updates.items() if not value.active]
        for key in terminal[:-32]:
            del self._updates[key]

    @staticmethod
    def _set_bar(bar, completed, total, active):
        value, maximum = bounded_bar_counts(completed, total)
        if not maximum and not active:
            maximum = 1  # Stop animation after failure/cancellation.
        bar.setRange(0, maximum)
        bar.setValue(value)
