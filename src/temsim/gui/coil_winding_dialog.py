"""Explicit winding inputs and read-only packing/current previews.

The caller commits one validated document transaction. This dialog neither
changes a live instrument nor builds CAD meshes or samples magnetic fields.
"""
from __future__ import annotations

from copy import deepcopy
import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit,
    QPlainTextEdit, QVBoxLayout,
)
from temsim.coil_winding import WIRE_AREA_FIELD, resolve_wire_cross_section_mm2, resolve_wire_diameter_mm


CURRENT_FIELD = "ideal_coil_current_a_at_100_percent"
INTERVAL_TURNS_FIELD = "winding_interval_turn_counts"


def _label(text=""):
    result = QLabel(text)
    result.setWordWrap(True)
    result.setTextFormat(Qt.TextFormat.PlainText)
    result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse
                                   | Qt.TextInteractionFlag.TextSelectableByKeyboard)
    return result


def _number_text(value):
    # Reopening and applying a saved calibration must preserve binary64 inputs.
    return "" if value is None else format(value, ".17g")


def _finite_number(text, label, *, positive=False):
    try:
        value = float(text)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(value) or value < 0 or positive and value == 0:
        requirement = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be finite and {requirement}")
    return value


class WindingConfigurationDialog(QDialog):
    """Edit fixed turns/current and optional conductor CAD dimensions."""

    def __init__(self, part, owner, by_key, *, runtime_values=None, commit=None, parent=None):
        super().__init__(parent)
        from temsim.cad_round_winding import round_winding_envelope

        self.setObjectName("windingConfigurationDialog")
        self.setWindowTitle("Configure winding")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(620, 620)
        self.part, self.owner = deepcopy(part), deepcopy(owner)
        self.by_key = deepcopy(by_key)
        self.runtime_values = deepcopy(runtime_values or {})
        self._commit = commit
        self._values = None
        self._geometry = None
        self.envelope = round_winding_envelope(self.part, self.by_key)

        layout = QVBoxLayout(self)
        self.heading = _label(f"{self.part.get('name', self.part['key'])}\nCurrent owner: {self.owner.get('name', self.owner['key'])}")
        layout.addWidget(self.heading)
        intervals = "; ".join(f"{a:g} to {b:g} mm" for a, b in self.envelope["intervals_mm"])
        self.envelope_label = _label(
            f"Existing field envelope (read only): ID {self.envelope['inner_diameter_mm']:g} mm; "
            f"OD {self.envelope['outer_diameter_mm']:g} mm.\nModule-local current-bearing Z intervals: {intervals}.")
        self.envelope_label.setObjectName("windingEnvelope")
        layout.addWidget(self.envelope_label)

        self.turn_count = QLineEdit()
        self.turn_count.setObjectName("windingTurnCount")
        self.turn_count.setPlaceholderText("Not specified; enter a positive integer")
        self.wire_cross_section = QLineEdit()
        self.wire_cross_section.setObjectName("windingWireCrossSection")
        self.wire_cross_section.setPlaceholderText("Optional; leave blank if conductor metal area is unknown")
        self.wire_diameter = _label()
        self.wire_diameter.setObjectName("derivedWindingWireDiameter")
        self.interval_turn_counts = QLineEdit()
        self.interval_turn_counts.setObjectName("windingIntervalTurnCounts")
        self.interval_turn_counts.setPlaceholderText("Optional; one fixed integer per Z interval, separated by commas")
        saved_intervals = self.part.get(INTERVAL_TURNS_FIELD)
        if saved_intervals is not None:
            self.interval_turn_counts.setText(", ".join(str(value) for value in saved_intervals))
        self.current_at_100_percent = QLineEdit()
        self.current_at_100_percent.setObjectName("windingCurrentAt100Percent")
        self.current_at_100_percent.setPlaceholderText("Not specified; enter conductor current in A")
        saved_turns = self.part.get("winding_turn_count")
        self.turn_count.setText("" if saved_turns is None else str(saved_turns))
        self._initial_area_text = _number_text(resolve_wire_cross_section_mm2(self.part))
        self.wire_cross_section.setText(self._initial_area_text)
        self.current_at_100_percent.setText(_number_text(self.owner.get(CURRENT_FIELD)))
        form = QFormLayout()
        form.addRow("Conductor turn count", self.turn_count)
        form.addRow("Fixed turns per axial interval", self.interval_turn_counts)
        show_intervals = len(self.envelope["intervals_mm"]) > 1
        self.interval_turn_counts.setVisible(show_intervals)
        form.labelForField(self.interval_turn_counts).setVisible(show_intervals)
        form.addRow("Conductor metal cross-section (mm²)", self.wire_cross_section)
        form.addRow("Derived round-wire diameter", self.wire_diameter)
        form.addRow("Owner current at 100% excitation (A)", self.current_at_100_percent)
        layout.addLayout(form)

        legacy = self.owner.get("air_core_reference_ampere_turns")
        self.legacy_label = _label(
            (f"Saved legacy reference: {legacy:g} A·turn. " if isinstance(legacy, (int, float)) else "")
            + "No historical turn count or wire diameter is inferred. In the ideal prescribed-current model, "
            "applying explicit turns/current replaces the legacy ampere-turn fallback; the saved reference is retained.")
        layout.addWidget(self.legacy_label)
        self.operating_label = _label()
        self.operating_label.setObjectName("windingActualCurrent")
        layout.addWidget(self.operating_label)
        self.packing = QPlainTextEdit()
        self.packing.setObjectName("windingPackingReadback")
        self.packing.setReadOnly(True)
        self.packing.setMinimumHeight(160)
        layout.addWidget(self.packing, 1)
        layout.addWidget(_label(
            "Physical turn counts are retained when current changes. A supplied conductor metal cross-section enables "
            "a uniform orthogonal wire grid; round-wire diameter is derived automatically. With unknown metal area, "
            "envelope CAD is retained without inferred wire geometry. When turns or metal area change, "
            "the remaining allocated winding volume is filled automatically by insulation, within the same ID, OD and axial envelope; "
            "existing passive layers remain separate. The ideal prescribed-current field keeps the same full "
            "envelope and uses turns times current; it is not compressed to the conductor area. "
            "Other field selections retain their own current and calibration rules. "
            "No lead geometry, resistance, heating or wire current rating is inferred."))
        self.validation = _label()
        self.validation.setObjectName("windingValidation")
        layout.addWidget(self.validation)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        self.apply_button = self.buttons.button(QDialogButtonBox.StandardButton.Apply)
        self.apply_button.setText("Apply to draft")
        self.apply_button.clicked.connect(self._apply)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        for control in (self.turn_count, self.interval_turn_counts, self.wire_cross_section, self.current_at_100_percent):
            control.textChanged.connect(self._refresh)
        self._refresh()

    def values(self):
        if self._values is None:
            raise ValueError("Enter and validate a complete winding configuration")
        return deepcopy(self._values)

    def set_runtime_values(self, values):
        self.runtime_values = deepcopy(values or {})
        self._refresh()

    def _read_values(self):
        text = self.turn_count.text().strip()
        try:
            turns = int(text)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Conductor turn count must be a positive integer") from error
        if turns <= 0:
            raise ValueError("Conductor turn count must be a positive integer")
        area_text = self.wire_cross_section.text().strip()
        result = dict(winding_turn_count=turns,
            ideal_coil_current_a_at_100_percent=_finite_number(self.current_at_100_percent.text().strip(), "Reference conductor current"))
        if area_text and area_text == self._initial_area_text:
            # Viewing a legacy diameter through its area readback is not an
            # edit or a reason to roundtrip its exact original numeric input.
            for name in (WIRE_AREA_FIELD, "winding_wire_diameter_mm"):
                if name in self.part:
                    result[name] = self.part[name]
        else:
            result[WIRE_AREA_FIELD] = (_finite_number(area_text, "Conductor metal cross-section", positive=True)
                                       if area_text else None)
            result["winding_wire_diameter_mm"] = None
        interval_text = self.interval_turn_counts.text().strip()
        if len(self.envelope["intervals_mm"]) == 1:
            if INTERVAL_TURNS_FIELD in self.part:
                result[INTERVAL_TURNS_FIELD] = [turns]
        elif interval_text:
            try:
                counts = [int(value.strip()) for value in interval_text.split(",")]
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("Fixed interval turns must be comma-separated positive integers") from error
            if len(counts) != len(self.envelope["intervals_mm"]) or any(value <= 0 for value in counts):
                raise ValueError("Supply one positive turn count for every occupied axial interval")
            if sum(counts) != turns:
                raise ValueError("Fixed interval turn counts must sum to the conductor turn count")
            result[INTERVAL_TURNS_FIELD] = counts
        elif INTERVAL_TURNS_FIELD in self.part:
            # Clearing this optional input is explicit; a total-N edit never
            # silently repartitions an existing fixed physical turn count.
            result[INTERVAL_TURNS_FIELD] = None
        try:
            finite_ni = math.isfinite(turns*result[CURRENT_FIELD])
        except OverflowError:
            finite_ni = False
        if not finite_ni:
            raise ValueError("Winding ampere-turns exceed the finite numerical range")
        return result

    def _operating_readback(self, values):
        percent = self.runtime_values.get("percent", self.runtime_values.get("excitation_percent"))
        polarity = self.runtime_values.get("polarity")
        if isinstance(percent, bool) or not isinstance(percent, (int, float)) or not math.isfinite(percent):
            return "Actual current readback unavailable: current operating excitation is not connected."
        if not 0 <= percent <= 100:
            return "Actual current readback unavailable: operating excitation is outside 0 to 100%."
        prefix = f"Operating excitation: {percent:g}%. "
        if values is None:
            return prefix+"Enter the reference conductor current to calculate the actual current."
        scaled = values[CURRENT_FIELD]*(float(percent)*.01)
        ni100 = values["winding_turn_count"]*values[CURRENT_FIELD]
        if self.runtime_values.get("enabled") is False:
            return (prefix+"Excitation disabled; signed conductor current 0 A.\n"
                f"Selected winding reference: {ni100:.9g} A·turn at 100% excitation.")
        if polarity in (-1, 1) and not isinstance(polarity, bool):
            signed = scaled*polarity
            return (prefix+f"Polarity {polarity:+g}; signed conductor current {signed:.9g} A.\n"
                f"Selected winding: {ni100:.9g} A·turn at 100%; {values['winding_turn_count']*signed:.9g} A·turn at this setting.")
        return (prefix+f"Conductor current magnitude {abs(scaled):.9g} A; polarity readback is unavailable.\n"
            f"Selected winding reference: {ni100:.9g} A·turn at 100% excitation.")

    @staticmethod
    def _packing_text(geometry):
        record = geometry.layout.to_metadata()
        envelope = geometry.field_envelope
        field_area = (envelope["outer_diameter_mm"]-envelope["inner_diameter_mm"])/2*sum(
            b-a for a,b in envelope["intervals_mm"])
        lines = [f"Conductor turns: {record['turn_count']}",
            f"Conductor metal cross-section per turn: {math.pi*record['wire_diameter_mm']**2/4:.9g} mm²",
            f"Ideal field equivalent cross-section (whole envelope): {field_area:.9g} mm²",
            f"Total conductor cross-section: {record['conductor_cross_section_area_mm2']:.9g} mm²",
            f"Uniform grid centre pitch: {record['centre_pitch_mm']:.9g} mm",
            f"Interwire insulation gap: {record['interwire_gap_mm']:.9g} mm",
            f"Conductor volume: {record['conductor_volume_mm3']:.9g} mm³",
            f"Remaining insulation volume: {record['insulation_volume_mm3']:.9g} mm³",
            f"Conductor volume fill: {record['volume_fill_fraction']*100:.6g}%"]
        for index, allocation in enumerate(record["allocations"], 1):
            lines.append(f"Allocation {index}: {allocation['turn_count']} turns; "
                f"{allocation['radial_rows']} radial × {allocation['axial_rows']} axial grid rows.")
        if geometry.passive_solids:
            lines.append("Retained passive regions: "+", ".join(spec.region for spec in geometry.passive_solids))
        return "\n".join(lines)

    def _refresh(self):
        from temsim.cad_round_winding import resolve_round_winding_geometry

        self._values = self._geometry = None
        try:
            text = self.wire_cross_section.text().strip()
            area = _finite_number(text, "Conductor metal cross-section", positive=True) if text else None
            diameter = resolve_wire_diameter_mm({WIRE_AREA_FIELD:area}) if area is not None else None
            self.wire_diameter.setText("Unknown; no wire geometry inferred" if diameter is None else f"{diameter:.9g} mm (calculated)")
        except ValueError:
            self.wire_diameter.setText("Unavailable until the metal cross-section is valid")
        missing = any(not control.text().strip() for control in (self.turn_count, self.current_at_100_percent))
        if missing:
            self.packing.setPlainText("Enter physical turns and reference current. Conductor metal cross-section is optional.")
            self.validation.setText("Enter turn count and reference current. No values are guessed.")
            self.validation.setStyleSheet("")
        else:
            try:
                values = self._read_values()
                candidate = deepcopy(self.part)
                for name in ("winding_turn_count", WIRE_AREA_FIELD, "winding_wire_diameter_mm", INTERVAL_TURNS_FIELD):
                    if name in values:
                        if values[name] is None:
                            candidate.pop(name, None)
                        else:
                            candidate[name] = deepcopy(values[name])
                geometry = resolve_round_winding_geometry(candidate, self.by_key)
                if geometry is None:
                    envelope = self.envelope
                    area = (envelope["outer_diameter_mm"]-envelope["inner_diameter_mm"])/2*sum(
                        b-a for a,b in envelope["intervals_mm"])
                    lines = [f"Conductor turns: {values['winding_turn_count']}",
                        f"Ideal field equivalent cross-section (whole envelope): {area:.9g} mm²",
                        "Conductor metal cross-section is unknown; envelope CAD is retained.",
                        "Detailed conductor layout, metal area and insulation fraction are unavailable.",
                        "The ideal field uses the declared physical turns and current without guessing conductor size."]
                    if values.get(INTERVAL_TURNS_FIELD) is not None:
                        lines.append("Fixed turns per axial interval: "+", ".join(
                            str(value) for value in values[INTERVAL_TURNS_FIELD]))
                    self.packing.setPlainText("\n".join(lines))
                else:
                    self.packing.setPlainText(self._packing_text(geometry))
                self._values, self._geometry = values, geometry
                self.validation.setText("Configuration is valid for the current draft. Apply stages one undoable change; Save remains separate.")
                self.validation.setStyleSheet("")
            except (ValueError, TypeError, OverflowError) as error:
                self.packing.setPlainText("The previous draft remains unchanged.")
                self.validation.setText(str(error))
                self.validation.setStyleSheet("color: #c65b5b")
        self.operating_label.setText(self._operating_readback(self._values))
        self.apply_button.setEnabled(self._values is not None)

    def _apply(self):
        if self._values is None:
            return
        try:
            if self._commit is not None:
                self._commit(self.values())
        except (ValueError, TypeError, RuntimeError) as error:
            self.validation.setText(str(error))
            self.validation.setStyleSheet("color: #c65b5b")
            return
        self.accept()
