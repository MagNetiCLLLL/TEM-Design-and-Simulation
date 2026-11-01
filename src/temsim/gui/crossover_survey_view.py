"""Read-only display of a saved unified particle crossover survey."""
from pathlib import Path
import json
import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)


def _number(value):
    if value is None:
        return "—"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Invalid numerical survey value")
    return f"{value:.8g}"


def _survey_rows(report):
    if report["schema"] != "unified-crossover-geometric-cs-v1":
        raise ValueError("Expected a unified particle crossover survey report")
    rows = []
    def measured(label, bracket, measurement):
        fits = measurement["pencils"]
        if len(fits) != 2:
            raise ValueError("Expected two reported pencil radii")
        rows.append((label, _number(measurement["z_mm"]), bracket,
                     *(_number(fit["c3_mm"]) for fit in fits),
                     " / ".join(fit["status"] for fit in fits)))
    for interval in report["lens_intervals"]:
        lens = interval["upstream_reference"]
        for waist in interval["regions"]:
            measured(lens, " to ".join(_number(v) for v in waist["bracket_mm"]),
                     waist["geometric_aberrations"])
        if not interval["regions"] or interval["status"] == "UNRESOLVED":
            status = interval["status"]
            message = ("No bounded 2D waist at this sampling" if status == "RESOLVED_AT_BOUNDED_SAMPLING"
                       else status)
            rows.append((lens, "—", "—", "—", "—", message))
    measured("Specimen target", "Fixed plane", report["target_aberrations"])
    return rows


class CrossoverSurveyView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        button = QPushButton("Open particle survey report…")
        button.clicked.connect(self._choose)
        layout.addWidget(button)
        self.summary = QLabel("Open the report.json produced by measure_crossover_cs.py.")
        self.summary.setTextFormat(Qt.TextFormat.PlainText)
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(("Lens interval / plane", "Z (mm)", "2D waist bracket (mm)",
                                              "Cs, full pupil (mm)", "Cs, small pupil (mm)", "Fit status"))
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)

    def _choose(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open saved crossover survey", "", "Survey JSON (*.json)")
        if filename:
            self.load_report(Path(filename))

    def load_report(self, path):
        self.table.setRowCount(0)
        try:
            if path.stat().st_size > 128*1024**2:
                raise ValueError("Survey JSON exceeds the viewer's 128 MiB limit")
            report = json.loads(path.read_text(encoding="utf-8"))
            rows = _survey_rows(report)
            radii = ", ".join(_number(r) for r in report["pupil"]["radii_um"])
            text = (f"Saved particle survey · {path}\nInput {report['input_digest']}\n"
                    f"Cumulative geometric Cs at each plane. Launch radii {radii} µm. "
                    "A dash denotes insufficient evidence or an unobserved plane.")
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            self.summary.setText(f"Unable to load survey. {exc}")
            return
        self.summary.setText(text)
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                self.table.setItem(i, j, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()

