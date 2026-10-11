"""System and component aberration diagnostics."""

from __future__ import annotations

from threading import Event
import numpy as np
import pyqtgraph as pg

from temsim.gui.input_policy import (
    WheelSafeComboBox as QComboBox,
)

from PySide6.QtCore import QObject, QRunnable, Qt, Signal, Slot
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from temsim.optics.aberrations import (
    SYSTEM_COEFFICIENT_ROWS,
    effective_aberration_comparison,
    _aberration_cache_signature,
)
from temsim.physics.beam_statistics import branch_sample_statistics
from temsim.gui.job_coordinator import CoordinatedPool, ResourceClaim
from temsim.optics.actual_probe_diagnostics import calculate_actual_probe_diagnostic, SCOPE


class _ActualSignals(QObject):
    solved = Signal(int, object)
    failed = Signal(int, str)
    finished = Signal(int)


class _ActualWorker(QRunnable):
    def __init__(self, generation, result):
        super().__init__()
        self.generation = generation
        self.existing_result = result
        self.event = Event()
        self.signals = _ActualSignals()
        self.resource_claim = ResourceClaim(2*1024**3, 512*1024**2)
        self.job_input_identity = f"actual-probe:{generation}:{getattr(result, 'signatures', {}).get('incident', '')}"

    def run(self):
        try:
            report = calculate_actual_probe_diagnostic(self.existing_result, cancelled=self.event.is_set)
            if not self.event.is_set():
                self.signals.solved.emit(self.generation, report)
        except Exception as exc:
            if not self.event.is_set():
                self.signals.failed.emit(self.generation, str(exc))
        finally:
            self.signals.finished.emit(self.generation)


class AberrationComparisonView(QWidget):
    """Display effective coefficients before and after active correction."""

    def __init__(self, parent=None, *, fixed_system: str | None = None) -> None:
        super().__init__(parent)
        if fixed_system not in {None, "probe", "image"}:
            raise ValueError("fixed_system must be probe, image or None")
        self.fixed_system = fixed_system
        self._state = None
        self._probe_statistics = None
        self._probe_statistics_error = "Run a calculation to measure the sample probe."
        self._stale = False
        self._result = None
        self._actual_report = None
        self._actual_worker = None
        self._generation = 0
        self._closed = False
        self.pool = CoordinatedPool(self)
        self.system = QComboBox()
        self.system.setObjectName("aberrationSystemSelector")
        self.system.addItem("Probe / specimen", "probe")
        self.system.addItem("Objective / image", "image")
        if fixed_system is not None:
            self.system.setCurrentIndex(self.system.findData(fixed_system))
            self.system.setEnabled(False)
        self.system.currentIndexChanged.connect(self._refresh)
        self.summary = QLabel(
            "Run a calculation to evaluate the active round-lens and "
            "multipole configuration."
        )
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.summary.setStyleSheet("font-weight: 600;")
        self.probe_summary = QLabel()
        self.probe_summary.setObjectName("sampleProbeShapeSummary")
        self.probe_summary.setWordWrap(True)
        self.probe_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.probe_summary.setToolTip(
            "Current-weighted geometric rays at the physical sample plane, using "
            "all surviving rays. D95 contains 95% of surviving current. Shape "
            "moments are |<z^n>|/<|z|^n> for centered z=x+iy, n=2 and 3, "
            "in [0, 1]. A three-lobed spot can have zero twofold moment. "
            "These are not fitted A1/A2 coefficients or wave intensity; small "
            "moments alone do not establish an ideal round probe. Finite-source "
            "sampling also contributes to the measured moments. Waist offset "
            "is a local linear extrapolation; positive means downstream."
        )
        self.table = QTableWidget(0, 7)
        self.table.setObjectName("aberrationComparisonTable")
        self.table.setHorizontalHeaderLabels(
            ("Term", "Meaning", "Reference", "Active model", "Difference", "Azimuth", "Basis")
        )
        for column, width in enumerate((48, 180, 100, 110, 110, 100)):
            self.table.setColumnWidth(column, width)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().hide()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        selector = QHBoxLayout()
        selector.addWidget(QLabel("Effective system"))
        selector.addWidget(self.system)
        selector.addStretch(1)
        reference_page = QWidget()
        reference = QVBoxLayout(reference_page)
        reference.addLayout(selector)
        reference.addWidget(self.summary)
        reference.addWidget(self.probe_summary)
        reference.addWidget(self.table, 1)
        self.pages = QTabWidget()
        self.pages.addTab(reference_page, "Reference coefficients")
        self.actual_page = QWidget()
        actual = QVBoxLayout(self.actual_page)
        actions = QHBoxLayout()
        self.actual_button = QPushButton("Measure actual incident beam")
        self.actual_button.setObjectName("measureActualIncidentBeam")
        self.actual_button.setProperty("calculationAction", True)
        self.actual_button.setEnabled(False)
        self.actual_button.clicked.connect(self._start_actual)
        self.actual_cancel = QPushButton("Cancel")
        self.actual_cancel.setEnabled(False)
        self.actual_cancel.clicked.connect(self._cancel_actual)
        actions.addWidget(self.actual_button)
        actions.addWidget(self.actual_cancel)
        actions.addStretch(1)
        actual.addLayout(actions)
        self.actual_summary = QLabel("Calculate rays first. This diagnostic uses the captured result, not unsaved live edits.")
        self.actual_summary.setWordWrap(True)
        self.actual_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.actual_summary.setToolTip(SCOPE)
        actual.addWidget(self.actual_summary)
        self.actual_table = QTableWidget(0, 7)
        self.actual_table.setHorizontalHeaderLabels(("Plane", "Z (mm)", "Rays", "RMS radius (nm)",
            "Local HP offset (µm)", "Non-affine RMS (nm)", "Entrance rank / 4"))
        self.actual_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.actual_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.actual_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.actual_table.horizontalHeader().setStretchLastSection(True)
        self.actual_table.setColumnWidth(0, 235)
        self.actual_table.currentCellChanged.connect(self._plot_actual)
        actual.addWidget(self.actual_table)
        plots = QHBoxLayout()
        self.actual_plot = pg.PlotWidget(title="Actual centred beam")
        self.residual_plot = pg.PlotWidget(title="After best affine entrance map")
        for plot in (self.actual_plot, self.residual_plot):
            plot.setLabel("bottom", "X", units="nm")
            plot.setLabel("left", "Y", units="nm")
            plot.setAspectLocked(True)
            plots.addWidget(plot, 1)
        actual.addLayout(plots, 1)
        self.actual_scope = QLabel(SCOPE)
        self.actual_scope.setWordWrap(True)
        self.actual_scope.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        actual.addWidget(self.actual_scope)
        self.pages.addTab(self.actual_page, "Actual incident beam")
        self.pages.setTabVisible(1, fixed_system != "image")
        layout = QVBoxLayout(self)
        layout.addWidget(self.pages)
        self._refresh_probe_summary()

    def display_result(self, result) -> None:
        if self._closed:
            return
        self._generation += 1
        self._cancel_actual()
        self._result = result
        self._actual_report = None
        self.actual_table.setRowCount(0)
        self.actual_plot.clear()
        self.residual_plot.clear()
        self.actual_summary.setText("Captured result ready. Measure actual HP passage and specimen beam; live edits are not used.")
        self.actual_button.setEnabled(self._actual_worker is None)
        self._state = getattr(result, "state_snapshot", None)
        self._probe_statistics = None
        try:
            self._probe_statistics = branch_sample_statistics(result.simulation.incident)
        except (AttributeError, ValueError) as exc:
            self._probe_statistics_error = f"Sample probe unavailable: {exc}"
        self._refresh_probe_summary()
        self._stale = True
        self.summary.setText(
            "Aberration state updated. Open this page to inspect coefficients."
        )

    def _refresh_probe_summary(self) -> None:
        self.pages.setTabVisible(1, str(self.fixed_system or self.system.currentData()) == "probe")
        self.probe_summary.setVisible(
            str(self.fixed_system or self.system.currentData()) == "probe"
        )
        stats = self._probe_statistics
        if stats is None:
            self.probe_summary.setText(self._probe_statistics_error)
            return
        self.probe_summary.setText(
            f"Sample-plane geometric probe | RMS radius {stats.radius_rms_m * 1.0e9:.4g} nm "
            f"| D95 {2.0 * stats.radius_95_m * 1.0e9:.4g} nm\n"
            f"Twofold moment {stats.twofold_moment:.4f} | threefold moment "
            f"{stats.threefold_moment:.4f} | waist offset "
            f"{stats.waist_offset_m * 1.0e9:+.4g} nm | "
            f"{stats.surviving_rays} surviving rays"
        )

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if self._stale:
            self._refresh()

    def _cancel_actual(self):
        if self._actual_worker is not None:
            self._actual_worker.event.set()
            self.actual_summary.setText("Cancelling actual-beam diagnostic…")
        self.pool.clear()

    def cancel_actual_diagnostic(self):
        """Public hook for the application's shared Cancel calculations action."""
        self._cancel_actual()

    def _start_actual(self):
        if self._closed or self._result is None or self._actual_worker is not None:
            return
        self._generation += 1
        worker = _ActualWorker(self._generation, self._result)
        self._actual_worker = worker
        worker.signals.solved.connect(self._actual_solved)
        worker.signals.failed.connect(self._actual_failed)
        worker.signals.finished.connect(self._actual_finished)
        self.actual_button.setEnabled(False)
        self.actual_cancel.setEnabled(True)
        self.actual_summary.setText("Continuing the captured entrance particles through HP2, HP1 and the specimen…")
        try:
            self.pool.start(worker)
        except Exception as exc:
            self._actual_worker = None
            self.actual_button.setEnabled(True)
            self.actual_cancel.setEnabled(False)
            self.actual_summary.setText("Actual incident-beam diagnostic could not start: " + str(exc))

    @Slot(int, object)
    def _actual_solved(self, generation, report):
        if (self._closed or generation != self._generation or self._actual_worker is None
                or self._actual_worker.event.is_set()):
            return
        self._actual_report = report
        p = report.provenance
        backends = sorted({r["actual"] for r in p["backend_receipts"] if r.get("outcome") == "returned"})
        replay_delta = p.get("retained_sample_difference_rms_m")
        replay_text = (f" Sample replay difference from retained rays: {replay_delta*1e9:.3g} nm RMS."
                       if replay_delta is not None else "")
        self.actual_summary.setText(
            f"Captured result {p['incident_signature'][:12]} | actual entrance Z {p['boundary_z_mm']:.8g} mm | "
            f"{p['entrance_surviving_rays']} of {p['source_ray_count']} rays | "
            f"{', '.join(backends) or 'backend not reported'} | {p['elapsed_s']:.3g} s\n"
            "Original energies, times and source weights retained." + replay_text
            + " Select a plane to inspect its centred beam and affine residual.")
        self.actual_table.setRowCount(len(report.planes))
        for index, row in enumerate(report.planes):
            stats, fit, local = row["statistics"], row["affine"], row["local_centroid_xy_m"]
            values = (row["label"], f"{row['z_mm']:.9g}", str(stats["surviving_rays"]) if stats else "0",
                f"{stats['radius_rms_m']*1e9:.5g}" if stats else "—",
                f"{local[0]*1e6:+.4g}, {local[1]*1e6:+.4g}" if local is not None else "—",
                f"{fit['rms_m']*1e9:.5g}" if fit["rms_m"] is not None else "—", str(fit["rank"]))
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(fit["status"] if col >= 5 else value)
                self.actual_table.setItem(index, col, item)
        self.actual_table.selectRow(len(report.planes)-1)

    @Slot(int, str)
    def _actual_failed(self, generation, message):
        if (not self._closed and generation == self._generation and self._actual_worker is not None
                and not self._actual_worker.event.is_set()):
            self.actual_summary.setText("Actual incident-beam diagnostic unavailable: " + message)

    @Slot(int)
    def _actual_finished(self, generation):
        if self._actual_worker is not None and self._actual_worker.generation == generation:
            cancelled = self._actual_worker.event.is_set()
            self._actual_worker = None
            self.actual_cancel.setEnabled(False)
            self.actual_button.setEnabled(not self._closed and self._result is not None)
            if cancelled and generation == self._generation and not self._closed:
                self.actual_summary.setText("Actual-beam diagnostic cancelled. Previous completed data, if any, remain visible.")

    def _plot_actual(self, row, *_args):
        report = self._actual_report
        self.actual_plot.clear()
        self.residual_plot.clear()
        if report is None or not 0 <= row < len(report.planes):
            return
        item = report.planes[row]
        stats = item["statistics"]
        if stats is None:
            return
        i = item["checkpoint_index"]
        mask = report.alive[i] & (report.weights > 0)
        x, y = report.checkpoints.x_m[i], report.checkpoints.y_m[i]
        self.actual_plot.plot((x[mask]-stats["mean_x_m"])*1e9, (y[mask]-stats["mean_y_m"])*1e9,
            pen=None, symbol="o", symbolSize=3, symbolBrush=(70, 200, 240, 160))
        residual = report.affine_residuals_m[row][mask]*1e9
        residual = residual[np.isfinite(residual).all(axis=1)]
        self.residual_plot.plot(residual[:, 0], residual[:, 1], pen=None, symbol="o", symbolSize=3,
            symbolBrush=(250, 170, 80, 160))
        self.actual_plot.enableAutoRange()
        self.residual_plot.enableAutoRange()

    def shutdown(self):
        self._closed = True
        self._generation += 1
        self._cancel_actual()
        return self.pool.waitForDone(3000)

    @staticmethod
    def _meaning(term: str) -> str:
        return {
            "C1": "defocus",
            "A1": "two-fold astigmatism",
            "B2": "axial coma",
            "A2": "three-fold astigmatism",
            "C3": "third-order spherical",
            "S3": "star aberration",
            "A3": "four-fold astigmatism",
            "C5": "fifth-order spherical",
            "A5": "six-fold astigmatism",
            "Cc": "first-order chromatic",
        }[term]

    @staticmethod
    def _coefficient_text(value: float) -> str:
        return f"{float(value):+.6g} mm"

    def _refresh(self, *_args) -> None:
        self._refresh_probe_summary()
        if self._state is None:
            return
        self._stale = False
        system = str(self.fixed_system or self.system.currentData())
        try:
            field_mode = getattr(self._state, f"{system}_aberrations", {}).get("mode") == "field_derived"
            if field_mode:
                entry = getattr(self._state, "_effective_aberration_cache", {}).get(system)
                if entry is None or entry[0] != _aberration_cache_signature(self._state, system):
                    raise ValueError("Run a calculation to fit field-derived coefficients")
            before, after, diagnostics = effective_aberration_comparison(
                self._state,
                system,
            )
        except Exception as exc:
            self.summary.setText(f"Aberration comparison unavailable: {exc}")
            self.table.setRowCount(0)
            return
        self.table.setColumnHidden(2, field_mode)
        self.table.setColumnHidden(4, field_mode)
        if field_mode:
            holdout = diagnostics.get("holdout_rms_m")
            validation = f" | holdout RMS {holdout * 1e9:.4g} nm" if holdout is not None else ""
            support = diagnostics.get("field_support_status", "support not assessed")
            self.summary.setText(
                f"{after.reference_plane} | Field-derived | fit RMS "
                f"{diagnostics['fit_rms_m'] * 1e9:.4g} nm{validation} | {support}\n"
                "correction comparison not calculated"
            )
            detail_text = (
                f"{diagnostics['source']}. {diagnostics['diagnostic_scope']}. "
                f"Fit condition number: {diagnostics['fit_condition_number']:.4g}. "
                "Fit residual is model approximation error, not a corrected beam size."
                f" Semi-angle: {diagnostics.get('fit_semiangle_mrad', 'unknown')} mrad; "
                f"step: {diagnostics.get('fit_step_mm', 'unknown')} mm. "
                f"Cc energy convergence: {diagnostics.get('chromatic_convergence', {}).get('status', 'NOT_ASSESSED')}. "
                f"Unmapped lenses: {', '.join(diagnostics.get('unmapped_round_lenses', ()))}. "
                f"Not implemented: {', '.join(diagnostics.get('unimplemented_terms', ()))}."
            )
        else:
            ratio = float(diagnostics["c3_residual_ratio"])
            rms_before = float(diagnostics["ray_error_rms_before"]) * 1.0e9
            rms_after = float(diagnostics["ray_error_rms_after"]) * 1.0e9
            detail_text = (
                f"Reference: {before.reference_plane}. C3 residual ratio "
                f"{ratio:+.4g}; transverse ray-error RMS "
                f"{rms_before:.4g} → {rms_after:.4g} nm. "
                f"{diagnostics['source']}. {diagnostics['diagnostic_scope']} "
                "Values are a non-OEM principle model."
            )
            self.summary.setText(
                f"{before.reference_plane} | independent reference-ring C3 residual {ratio:+.4g} | "
                f"ray-error RMS {rms_before:.4g} → {rms_after:.4g} nm. "
                "This ring does not inherit the actual source beam. Other residual coefficients are not measured."
            )
        self.summary.setToolTip(detail_text)
        self.table.setRowCount(len(SYSTEM_COEFFICIENT_ROWS))
        for row, (term, value_name, angle_name) in enumerate(SYSTEM_COEFFICIENT_ROWS):
            value_before = float(getattr(before, value_name))
            value_after = float(getattr(after, value_name))
            unmeasured = term in diagnostics["unmeasured_coefficients"]
            values = (
                term,
                self._meaning(term),
                "—" if unmeasured else self._coefficient_text(value_before),
                "—" if unmeasured else self._coefficient_text(value_after),
                "—" if unmeasured else self._coefficient_text(value_after - value_before),
                (
                    "—" if unmeasured else
                    "axisymmetric"
                    if angle_name is None
                    else f"{float(getattr(after, angle_name)):.4g}°"
                ),
                diagnostics["coefficient_status"][term],
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                if 2 <= column <= 5:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.table.setItem(row, column, item)
