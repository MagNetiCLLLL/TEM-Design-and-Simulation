"""Bind current validation evidence to exact installed inputs and source bytes."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
VALIDATION = HERE / 'validation'


def digest(path):
    return dict(path=path.relative_to(ROOT).as_posix(),
        sha256=sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def main():
    inputs = json.loads((VALIDATION / 'input-equivalence.json').read_text(encoding='utf-8'))
    for row in inputs['installed_inputs']:
        assert sha256((ROOT / row['path']).read_bytes()).hexdigest() == row['sha256']
    peaks = json.loads((VALIDATION / 'independent-peak-field-checks.json').read_text(encoding='utf-8'))
    assert peaks['status'] == 'passed'
    xml = ET.parse(VALIDATION / 'final-regression-tests.xml').getroot()
    suites = list(xml.iter('testsuite'))
    counts = {key: sum(int(s.get(key, '0')) for s in suites)
        for key in ('tests', 'errors', 'failures', 'skipped')}
    cases = list(xml.iter('testcase'))
    identities = {(c.get('classname'), c.get('name')) for c in cases}
    assert counts['tests'] == len(cases) == len(identities)
    assert counts['errors'] == counts['failures'] == counts['skipped'] == 0
    current_names = (
        'input-equivalence.json', 'turn-initialization.json',
        'current-calibration-final.json', 'independent-peak-field-checks.json',
        'field-profile-data.json', 'peak-matched-field-profiles.png',
        'default-command-currents.json', 'all-installed-packing-bounds.json',
        'default-native-cad.json', 'default-native-cad-image_dph1_coils.json',
        'final-native-cad.json',
        'cad-electrical-refinement-equivalence.json', 'installed-fine-wire-cad.png',
        'transverse-step-input.json', 'transverse-step-receipt.json',
        'probe_dph2_coils-winding-dialog.png', 'condenser_stigmator-winding-dialog.png',
        'final-regression-tests.xml',
    )
    assert all((VALIDATION / name).is_file() for name in current_names)
    files = set((ROOT / 'src/temsim').rglob('*.py'))
    files.update((ROOT / 'tests').rglob('*.py'))
    test_start = min(datetime.fromisoformat(s.attrib['timestamp']).timestamp() for s in suites)
    assert all(p.stat().st_mtime <= test_start for p in files), 'Source/test bytes changed after the final run started'
    latest_source_mtime = max(p.stat().st_mtime for p in files)
    files.update(HERE.glob('*.py'))
    files.update(ROOT / r['path'] for r in inputs['installed_inputs'])
    files.update((HERE / 'README.md', ROOT / 'docs/TRANSVERSE_CURRENT_WINDINGS.md',
        ROOT / 'docs/SIMULATION_MODES.md', ROOT / 'docs/CALCULATION_SCOPE_MATRIX.md',
        ROOT / 'docs/IDEAL_CURRENT_WINDINGS.md', ROOT / 'agent.md', ROOT / '.gitattributes'))
    output_files = [p for p in VALIDATION.rglob('*') if p.is_file() and p.name != 'manifest.json']
    receipt = dict(schema='transverse-current-coils-evidence-manifest-v1',
        generated_utc=datetime.now(timezone.utc).isoformat(),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        working_tree_has_prior_uncommitted_changes=True,
        source_hash_scope='Full current Python source and tests are bound for dependency reproducibility; this is not an exclusive task diff',
        environment=dict(python=sys.version, platform=platform.platform(),
            packages={name: version(name) for name in ('numpy', 'scipy', 'pytest', 'PySide6', 'manifold3d')}),
        final_regression=counts, unique_test_identities=len(identities),
        source_and_tests_unmodified_after_test_start=True,
        latest_source_or_test_mtime_utc=datetime.fromtimestamp(latest_source_mtime, timezone.utc).isoformat(),
        current_evidence=list(current_names),
        earlier_receipt_policy='Other JUnit/calibration receipts and history/ are intermediate runs; before/ is the preserved premigration baseline',
        final_field_maximum_peak_normalized_error=peaks['maximum_peak_normalized_error'],
        finite_source_particle_or_coherent_execution=False,
        installed_inputs=inputs['installed_inputs'],
        source_files=[digest(p) for p in sorted(files)],
        evidence_files=[digest(p) for p in sorted(output_files)])
    destination = VALIDATION / 'manifest.json'
    destination.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(final_regression=counts, source_files=len(files),
        evidence_files=len(output_files), maximum_peak_error=peaks['maximum_peak_normalized_error'])))


if __name__ == '__main__':
    main()
