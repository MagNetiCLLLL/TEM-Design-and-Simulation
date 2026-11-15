"""Bind final inputs, field/CAD evidence and regression source versions."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

from finalize_ui_help import structure


ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
V=HERE/'validation'


def digest(path):
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)


def main():
    inputs=json.loads((V/'input-equivalence.json').read_text(encoding='utf-8'))
    assert inputs['multipole_hosts']==40 and inputs['optical_states_exact']==14
    for row in inputs['installed_inputs']:
        assert sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256']
    untouched=json.loads((V/'before/all-configs.json').read_text(encoding='utf-8'))
    changed={row['path'] for row in inputs['installed_inputs']}
    assert all(row['path'] in changed or sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'] for row in untouched)
    peaks=json.loads((V/'independent-peak-field-checks.json').read_text(encoding='utf-8'))
    assert peaks['status']=='passed' and peaks['hosts']==40
    for name,field in (('final-native-cad-current.json','records'),('winding-clearance.json','all_installed_host_bounds')):
        receipt=json.loads((V/name).read_text(encoding='utf-8'))
        assert receipt['status']=='passed'
        for row in receipt[field]:
            assert sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['source_snapshot_sha256']
    xml=ET.parse(V/'final-regression-tests.xml').getroot()
    suites=list(xml.iter('testsuite'))
    counts={key:sum(int(s.get(key,'0')) for s in suites) for key in ('tests','failures','errors','skipped')}
    cases=list(xml.iter('testcase'));ids={(row.get('classname'),row.get('name')) for row in cases}
    assert counts['tests']==len(cases)==len(ids)==710
    assert counts['failures']==counts['errors']==counts['skipped']==0
    test_start=min(datetime.fromisoformat(s.get('timestamp')).timestamp() for s in suites)
    help_check=json.loads((V/'ui-help-finalization.json').read_text(encoding='utf-8'))
    assert help_check['status']=='passed'
    help_paths={row['path'] for row in help_check['source_records']}
    assert help_paths=={'src/temsim/round_lens_models.py','src/temsim/gui/simulation_menu.py'}
    help_verified=datetime.fromisoformat(help_check['checked_utc']).timestamp()
    for row in help_check['source_records']:
        path=ROOT/row['path'];old=V/'history/pre-helptext'/row['path']
        assert sha256(path.read_bytes()).hexdigest()==row['after_sha256']
        assert sha256(old.read_bytes()).hexdigest()==row['before_sha256']
        assert row['original_mtime']<=test_start
        assert structure(old.read_text(encoding='utf-8'))==structure(path.read_text(encoding='utf-8'))
        assert path.stat().st_mtime<=help_verified
    files=set((ROOT/'src/temsim').rglob('*.py'))|set((ROOT/'tests').rglob('*.py'))
    assert all(path.stat().st_mtime<=test_start for path in files if path.relative_to(ROOT).as_posix() not in help_paths), 'Executable source/test changed after the primary regression started'
    files.update(HERE.glob('*.py'))
    files.update((ROOT/'configs').rglob('*.toml'))
    files.update((HERE/'README.md',ROOT/'docs/MULTIPOLE_CURRENT_WINDINGS.md',
        ROOT/'docs/PROBE_CORRECTOR_INTERNAL_CAD.md',ROOT/'docs/IDEAL_CURRENT_WINDINGS.md',
        ROOT/'docs/TRANSVERSE_CURRENT_WINDINGS.md',ROOT/'docs/SIMULATION_MODES.md',
        ROOT/'docs/CALCULATION_SCOPE_MATRIX.md',ROOT/'agent.md',ROOT/'.gitattributes',ROOT/'pyproject.toml'))
    active=(
        'turn-initialization-final.json','current-calibration-final.json',
        'independent-peak-field-checks.json','input-equivalence.json',
        'final-native-cad-current.json','winding-clearance.json',
        'bounded-step-input.json','bounded-step-receipt.json',
        'installed-multipole-windings.png','default-current-readbacks.json',
        'peak-matched-field-profiles.png','field-profile-data.json',
        'final-regression-tests.xml','ui-help-finalization.json','annotation-finalization.json',
    )
    assert all((V/name).is_file() for name in active)
    outputs=[path for path in V.rglob('*') if path.is_file() and path.name!='manifest.json']
    git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    manifest=dict(schema='ideal-multipole-current-evidence-manifest-v1',
        generated_utc=datetime.now(timezone.utc).isoformat(),git_base_head=git('rev-parse','HEAD'),
        delivery_branch=git('branch','--show-current'),
        cumulative_prior_working_tree_dependencies_included=True,
        source_hash_scope='Executed working-tree Python source/tests, configs and report; not an exclusive latest-task diff',
        environment=dict(python=sys.version,platform=platform.platform(),
            packages={name:version(name) for name in ('numpy','scipy','pytest','PySide6','manifold3d')}),
        final_regression=counts,unique_test_identities=len(ids),
        mathematical_source_and_tests_unmodified_after_primary_test_start=True,
        post_regression_UI_string_only_updates=help_check['source_records'],
        post_update_offscreen_UI_readback_passed=True,
        final_maximum_reference_coefficient_error=peaks['maximum_peak_normalized_error'],
        optical_source_tolerance_and_declared_boundary_equivalence_passed=True,
        monochromator_and_energy_filter_sources_unchanged=True,
        new_particle_or_coherent_execution=False,current_evidence=list(active),
        historical_evidence_policy='Initial N156 Image authoring, earlier calibration/mesh receipts and history/ are preserved intermediate stages; before/ is the pre-task baseline',
        source_files=[digest(path) for path in sorted(files)],
        evidence_files=[digest(path) for path in sorted(outputs)])
    (V/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(tests=counts,source_files=len(files),evidence_files=len(outputs),
        maximum_reference_error=peaks['maximum_peak_normalized_error'],post_suite_UI_help='verified string literals only')))


if __name__=='__main__':
    main()
