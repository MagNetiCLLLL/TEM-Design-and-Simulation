"""Explicit, cancellable assembly-error experiments over captured probe optics."""
from threading import Event

from PySide6.QtCore import QObject, QRunnable, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton, QSpinBox,
    QTableWidget, QVBoxLayout, QWidget,
)

from temsim.assembly_error_study import (
    AssemblyError, AssemblyErrorCase, AssemblyStudyOptions, THIRD_ORDER_NAMES,
    capture_study, run_study, save_study, supported_error_parts,
)
from temsim.gui.input_policy import WheelSafeComboBox, WheelSafeDoubleSpinBox
from temsim.gui.job_coordinator import CoordinatedPool, ResourceClaim
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.lens_pose import pose_control_label, pose_control_values
from temsim.optics.probe_service import ROUTES, service_settings


def _label(text):
    widget = QLabel(text)
    widget.setWordWrap(True)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse
                                   | Qt.TextInteractionFlag.TextSelectableByKeyboard)
    return widget


class _Signals(QObject):
    progress = Signal(int, str)
    ready = Signal(int, object)
    failed = Signal(int, str)
    finished = Signal(int)


class _Worker(QRunnable):
    def __init__(self, generation, request, live_identity):
        super().__init__()
        self.generation, self.request, self.live_identity = generation, request, live_identity
        self.cancellation = Event()
        self.signals = _Signals()
        self.resource_claim = ResourceClaim(2 * 1024**3, 512 * 1024**2)
        self.job_input_identity = request.digest

    def run(self):
        try:
            result = run_study(self.request, cancelled=self.cancellation.is_set,
                progress=lambda message: self.signals.progress.emit(self.generation, message))
            if not self.cancellation.is_set():
                self.signals.ready.emit(self.generation, result)
        except Exception as exc:
            if not self.cancellation.is_set():
                self.signals.failed.emit(self.generation, str(exc))
        finally:
            self.signals.finished.emit(self.generation)


def result_text(result):
    lines = ["Assembly error / aberration / correction headroom", result["scope"], "",
             "Nominal: captured instrument, not an assumed perfect assembly.",
             "Reported coefficient changes are relative to this nominal instrument.",
             "Electrical headroom uses model maxima and percentage points.", ""]
    limits = result["options"]
    lines.append(f"Requested tolerances: pupil RMS {limits['pupil_rms_tolerance_nm']:.6g} nm; "
                 f"chief shift {limits['centroid_tolerance_nm']:.6g} nm; direction "
                 f"{limits['direction_tolerance_mrad']:.6g} mrad.")
    for row in result["routes"]:
        lines.append(f"{row['route']}: intrinsic {dict(row['intrinsic'])}; learning {dict(row['learning'])}")
    for case in result["cases"]:
        lines.extend(("", case["name"] + " — " + case["status"]))
        for error in case["errors"]:
            lines.append(f"  {error['component']}.{error['parameter']}: delta {error['delta']:.8g}")
        if case["status"] == "FAILED":
            lines.append(case["failure"])
            continue
        nominal = result["nominal"]["coefficients_mm"]
        before = case["uncompensated"]["coefficients_mm"]
        lines.append("Uncompensated coefficient changes (mm): " + ", ".join(
            f"{key} {before[key]-nominal[key]:.7g}" for key in THIRD_ORDER_NAMES))
        for name in ("intrinsic_only", "with_learning"):
            stage = case[name]
            lines.append(f"{name.replace('_', ' ')}: {stage['status']}")
            if stage.get("failure"):
                lines.append("  " + stage["failure"])
            if "errors" not in stage:
                continue
            errors = stage["errors"]
            lines.append(f"  Target pupil RMS {errors['pupil_rms_nm']:.6g} nm; chief shift "
                         f"{errors['centroid_distance_nm']:.6g} nm; direction {errors['direction_distance_mrad']:.6g} mrad")
            lines.append("  Remaining coefficients (mm): " + ", ".join(
                f"{key} {errors['coefficient_delta_mm'][key]:.7g}" for key in THIRD_ORDER_NAMES))
            lines.append("  Route increments (% points): " + str(dict(stage["route_change_percent"])))
            if stage.get("reused_evidence"):
                lines.append("  " + stage["reused_evidence"])
            for output in stage["headroom_after"]["outputs"]:
                lines.append(f"  {output['output']}: maximum {output['maximum']:.6g} {output['unit']}; "
                             f"setting {output['percent']:.6g}%; remaining +{output['positive_margin_percent']:.6g} / "
                             f"-{output['negative_margin_percent']:.6g} percentage points")
    lines.extend(("", result["qualification"],
                  "Finite-source emitted / surviving / effective counts and current: NOT_ASSESSED.",
                  f"Diagnostic measurements: {result['measurement_count']}; elapsed {result['elapsed_s']:.2f} s"))
    return "\n".join(lines)


class AssemblyErrorStudyControls(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("assemblyErrorStudyControls")
        self._state = self._worker = self._saved = None
        self._generation = self._revision = 0
        self.pool = CoordinatedPool(self)
        self.pool.setMaxThreadCount(1)
        layout = QVBoxLayout(self)
        layout.addWidget(_label(
            "Compare assembly errors against the captured probe column using a fixed diagnostic pupil. "
            "Only the selected service inputs may compensate. Intrinsic cross-coupling remains physical; "
            "learning uses separately saved gains and is never inferred from a small response. "
            "Results include electrical headroom. Source transmission and physical qualification require separate studies."))
        self.errors = QTableWidget(0, 3)
        self.errors.setHorizontalHeaderLabels(("Placement owner", "Error parameter", "Delta"))
        self.errors.setColumnWidth(0, 240)
        self.errors.setColumnWidth(1, 340)
        self.errors.setMinimumHeight(130)
        self.errors.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.errors)
        row = QHBoxLayout()
        self.add_button, self.remove_button = QPushButton("Add error"), QPushButton("Remove selected error")
        self.add_button.clicked.connect(self.add_error)
        self.remove_button.clicked.connect(lambda: self.errors.removeRow(self.errors.currentRow()))
        row.addWidget(self.add_button)
        row.addWidget(self.remove_button)
        layout.addLayout(row)
        self.separate = QCheckBox("Also compare each error separately")
        layout.addWidget(self.separate)
        form = QFormLayout()
        self.routes = QListWidget()
        self.routes.setMaximumHeight(140)
        form.addRow(_label("Allowed service inputs"), self.routes)
        self.targets = QListWidget()
        self.targets.setMaximumHeight(110)
        for name in THIRD_ORDER_NAMES:
            item = QListWidgetItem(name, self.targets)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
        form.addRow(_label("Coefficient targets to restore"), self.targets)
        self.learning = QCheckBox("Compare with configured learning gains")
        self.learning.setToolTip("Uses saved gains even if live learning is off. Zero gains remain zero. No learning matrix is fitted or applied to the instrument.")
        form.addRow(self.learning)
        defaults = AssemblyStudyOptions()
        fields = (
            ("semiangle_mrad", "Diagnostic pupil", " mrad", 100.),
            ("maximum_step_mm", "Maximum axial step", " mm", 10.),
            ("calibration_step_percent", "Signed calibration increment", " % points", 100.),
            ("maximum_change_percent", "Total route search bound", " % points", 100.),
            ("pupil_rms_tolerance_nm", "Target pupil displacement RMS", " nm", 1e6),
            ("centroid_tolerance_nm", "Chief-position tolerance", " nm", 1e9),
            ("direction_tolerance_mrad", "Chief-direction tolerance", " mrad", 1e3),
        )
        self.editors = {}
        for name, title, suffix, maximum in fields:
            editor = WheelSafeDoubleSpinBox()
            editor.setDecimals(6)
            editor.setRange(.000001, maximum)
            editor.setValue(getattr(defaults, name))
            editor.setSuffix(suffix)
            editor.setKeyboardTracking(False)
            self.editors[name] = editor
            form.addRow(_label(title), editor)
        self.iterations = QSpinBox()
        self.iterations.setRange(1, 12)
        self.iterations.setValue(defaults.maximum_iterations)
        form.addRow(_label("Maximum local iterations"), self.iterations)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.run_button = QPushButton("Run captured assembly study")
        self.run_button.setProperty("calculationAction", True)
        self.cancel_button, self.save_button = QPushButton("Cancel"), QPushButton("Save study and report…")
        self.run_button.clicked.connect(self.start)
        self.cancel_button.clicked.connect(self.cancel)
        self.save_button.clicked.connect(self.save)
        for button in (self.run_button, self.cancel_button, self.save_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.status = _label("Add errors and select the permitted service inputs. The instrument is not changed by a study.")
        layout.addWidget(self.status)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMinimumHeight(240)
        layout.addWidget(self.report)
        self._buttons()

    def set_state(self, state):
        changed = state is not self._state
        if changed:
            self.cancel()
        self._state = state
        self.refresh_routes()
        if changed:
            self.errors.setRowCount(0)
            self.add_error()
        self._buttons()

    def refresh_routes(self):
        checked = {self.routes.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.routes.count())
                   if self.routes.item(i).checkState() == Qt.CheckState.Checked}
        if not self.routes.count():
            checked = set(AssemblyStudyOptions().routes)
        self.routes.clear()
        if self._state is None:
            return
        for key, route in ROUTES.items():
            primary, learning, _ = service_settings(self._state, key)
            item = QListWidgetItem(key, self.routes)
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setToolTip(f"Intrinsic: {primary}\nLearning: {learning}\n{route.evidence}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            if not any(primary.values()):
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setText(key + " (intrinsic gains required)")
            item.setCheckState(Qt.CheckState.Checked if key in checked and any(primary.values()) else Qt.CheckState.Unchecked)

    def set_revision(self, revision):
        if int(revision) != self._revision:
            self._revision = int(revision)
            self.cancel()
            self.refresh_routes()
            self.status.setText("Instrument inputs changed. Run a fresh study; any displayed report retains its captured inputs.")

    def add_error(self):
        if self._state is None:
            return
        parts = supported_error_parts(self._state)
        if not parts:
            return
        row = self.errors.rowCount()
        self.errors.insertRow(row)
        component, parameter = WheelSafeComboBox(), WheelSafeComboBox()
        for part in parts:
            component.addItem(part.key, part.key)
        index = component.findData("probe_hp1_hexapole")
        component.setCurrentIndex(max(0, index))
        delta = WheelSafeDoubleSpinBox()
        delta.setDecimals(6)
        delta.setRange(-1e6, 1e6)
        delta.setValue(.01)
        def update_parameters():
            old = parameter.currentData()
            parameter.clear()
            part = next(p for p in parts if p.key == component.currentData())
            for name in pose_control_values(part.data):
                if name == "rotation_z_mrad":
                    continue
                title, unit = pose_control_label(name)
                parameter.addItem(f"{title} ({unit})", name)
            preferred = old or "field_azimuth_error_deg"
            parameter.setCurrentIndex(max(0, parameter.findData(preferred)))
        component.currentIndexChanged.connect(update_parameters)
        update_parameters()
        for col, widget in enumerate((component, parameter, delta)):
            self.errors.setCellWidget(row, col, widget)

    def _capture(self):
        errors = tuple(AssemblyError(self.errors.cellWidget(row, 0).currentData(),
            self.errors.cellWidget(row, 1).currentData(), self.errors.cellWidget(row, 2).value())
            for row in range(self.errors.rowCount()))
        cases = [AssemblyErrorCase("Combined errors", errors)]
        if self.separate.isChecked() and len(errors) > 1:
            cases = [AssemblyErrorCase(f"Error {i + 1}: {error.component}.{error.parameter}", (error,))
                     for i, error in enumerate(errors)] + cases
        routes = tuple(self.routes.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.routes.count())
                       if self.routes.item(i).checkState() == Qt.CheckState.Checked)
        targets = tuple(self.targets.item(i).text() for i in range(self.targets.count())
                        if self.targets.item(i).checkState() == Qt.CheckState.Checked)
        options = AssemblyStudyOptions(routes=routes, target_coefficients=targets,
            compare_learning=self.learning.isChecked(), maximum_iterations=self.iterations.value(),
            **{name: widget.value() for name, widget in self.editors.items()})
        return capture_study(self._state, cases, options)

    def start(self):
        if self._state is None or self._worker is not None:
            return
        try:
            identity = capture_instrument_snapshot(self._state).physical_digest
            request = self._capture()
            self._generation += 1
            worker = _Worker(self._generation, request, identity)
            worker.signals.progress.connect(self._progress)
            worker.signals.ready.connect(self._ready)
            worker.signals.failed.connect(self._failed)
            worker.signals.finished.connect(self._finished)
            self._worker = worker
            self.status.setText("Running captured assembly study…")
            self._buttons()
            self.pool.start(worker)
        except Exception as exc:
            self._worker = None
            self.status.setText("Study could not start: " + str(exc))
            self._buttons()

    @Slot(int, str)
    def _progress(self, generation, message):
        if generation == self._generation and self._worker is not None:
            self.status.setText(message)

    @Slot(int, object)
    def _ready(self, generation, result):
        worker = self._worker
        if worker is None or generation != self._generation or worker.cancellation.is_set():
            return
        if capture_instrument_snapshot(self._state).physical_digest != worker.live_identity:
            self.cancel()
            self.status.setText("Inputs changed during the study; result was not installed. Run a fresh study.")
            return
        self._saved = (worker.request, result)
        self.report.setPlainText(result_text(result))
        failed = sum(row["status"] != "COMPLETE" for row in result["cases"])
        self.status.setText(f"Study finished: {len(result['cases'])} cases, {failed} with unavailable stages. Review residuals and model headroom below.")
        self._buttons()

    @Slot(int, str)
    def _failed(self, generation, message):
        if generation == self._generation:
            self.status.setText("Study unavailable: " + message)

    @Slot(int)
    def _finished(self, generation):
        if generation == self._generation:
            self._worker = None
            self._buttons()

    def _buttons(self):
        busy = self._worker is not None
        self.run_button.setEnabled(not busy and self._state is not None
                                   and bool(getattr(self._state, "probe_corrector_installed", False)))
        self.cancel_button.setEnabled(busy)
        self.save_button.setEnabled(self._saved is not None and not busy)
        for widget in (self.errors, self.routes, self.targets, self.add_button, self.remove_button,
                       self.separate, self.learning, self.iterations, *self.editors.values()):
            widget.setEnabled(not busy)

    def cancel(self):
        if self._worker is not None:
            self._worker.cancellation.set()
            self._generation += 1
            self._worker = None
            self.pool.clear()
            self.status.setText("Study cancelled. Captured inputs and instrument settings are retained.")
        self._buttons()

    def save(self):
        if self._saved is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save assembly study", "assembly-study.json", "JSON (*.json)")
        if path:
            try:
                save_study(path, *self._saved)
                self.status.setText("Saved captured inputs and study report: " + path)
            except (OSError, ValueError) as exc:
                self.status.setText("Could not save study: " + str(exc))

    def shutdown(self):
        self.cancel()
        return self.pool.waitForDone(3_000)
