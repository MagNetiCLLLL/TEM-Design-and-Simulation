"""Compare current fixed-N CAD inputs with the previous frozen field receipt.

No particles or coherent waves are calculated. The unchanged previous capture
routine is reused to reproduce six installed assembly cases, their complete
optical states, native multipole fields and all 31 material records per HP.
"""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
PREVIOUS = ROOT/'docs/development/probe_corrector_details_20261115'


def forbidden(*args, **kwargs):
    raise AssertionError('Field/CAD comparison must not emit or transport a source')


def main():
    capture_path = PREVIOUS/'validate_details.py'
    baseline_path = PREVIOUS/'validation/state-fields.json'
    spec = importlib.util.spec_from_file_location('previous_probe_detail_capture', capture_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit', forbidden), \
         patch('temsim.optics.electron_gun.source.trace_source_to_exit', forbidden):
        current = module.capture()
    assert current == baseline, 'Optics, native field, or original CAD changed'
    count = sum(len(case['fields']) for case in current['cases'])
    assert count == 78 and len(current['cases']) == 6
    receipt = dict(status='passed', assembly_cases=6, native_field_records_exact=count,
        nonzero_native_field_records=sum(any(abs(value)>0 for row in field['b_t'] for value in row)
            for case in current['cases'] for field in case['fields']),
        optical_state_exact=True, all_31_hp_solid_records_exact=True,
        source_or_wave_executed=False,
        baseline_path=baseline_path.relative_to(ROOT).as_posix(),
        baseline_sha256=sha256(baseline_path.read_bytes()).hexdigest(),
        capture_path=capture_path.relative_to(ROOT).as_posix(),
        capture_sha256=sha256(capture_path.read_bytes()).hexdigest())
    output=REPORT/'validation/native-field-equivalence.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
