"""Bind executed winding initialization evidence to source and original inputs."""
from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT=Path(__file__).resolve().parents[3]
REPORT=Path(__file__).resolve().parent
DESTINATION=REPORT/'validation/manifest.json'
CONFIGS=('C3_ProbeCorrector.toml', 'C3_ProbeCorrector_ImageCorrector.toml', 'C3_ProbeCorrector_BCOR.toml')
IMPLEMENTATION=(
    'src/temsim/cad_probe_winding.py', 'src/temsim/gui/probe_winding_dialog.py',
    'src/temsim/part_model_document.py', 'src/temsim/parameter_semantics.py',
    'src/temsim/parameter_impact.py', 'src/temsim/module_manifest.py',
    'src/temsim/probe_hexapole_geometry.py', 'src/temsim/cad_multipoles.py',
    'src/temsim/physics/instrument_magnetic.py', 'src/temsim/physics/posed_column_fields.py',
    'src/temsim/optics/stigmator_field.py', 'src/temsim/optics/corrector_channels.py',
    'configs/physics/coil_winding_estimates.toml',
    'tests/test_probe_winding_inputs.py', 'tests/test_corrector_turn_estimation.py',
    'tests/test_cad_probe_winding.py', 'tests/test_probe_winding_region_identity.py',
    'tests/test_winding_parameter_information.py', 'tests/test_coil_winding_dialog.py',
    'tests/test_part_model_document.py', 'tests/test_cad_multipoles.py',
    'tests/test_probe_hexapole_support.py',
)


def record(path):
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                sha256=sha256(path.read_bytes()).hexdigest())


def main():
    tests=[]
    case_ids=[]
    for path in sorted((REPORT/'validation').glob('*tests.xml')):
        root=ET.parse(path).getroot()
        for suite in root.iter('testsuite'):
            for field in ('failures','errors','skipped'):
                assert int(suite.attrib[field])==0
            tests.append(dict(file=path.name, tests=int(suite.attrib['tests']),
                              failures=0, errors=0, skipped=0))
        for case in root.iter('testcase'):
            case_ids.append((case.attrib.get('classname'),case.attrib['name']))
    assert len(case_ids)==len(set(case_ids))==sum(t['tests'] for t in tests)==294
    configs=[ROOT/'configs/instruments/column'/name for name in CONFIGS]
    originals=[REPORT/'validation/before'/name for name in CONFIGS]
    files=[path for path in REPORT.rglob('*') if path.is_file()
           and path!=DESTINATION and '__pycache__' not in path.parts and path.suffix!='.pyc']
    receipt=dict(schema='corrector-turn-current-study-evidence-v1', status='passed',
        scope='fixed-N initialization and bounded field/CAD/UI research; not experimental or beam qualification',
        installed_configs=[record(path) for path in configs],
        preserved_original_configs=[record(path) for path in originals],
        implementation=[record(ROOT/path) for path in IMPLEMENTATION],
        tests=tests, unique_passed_tests=len(case_ids),
        evidence=[record(path) for path in sorted(files)],
        default_optics_tolerances_and_native_field_providers_unchanged=True,
        particle_or_coherent_calculation_executed=False,
        research_coil_fields_installed_as_runtime_sources=False)
    DESTINATION.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(f'Bound {len(files)} evidence files and {len(case_ids)} unique passing tests: {DESTINATION}')


if __name__=='__main__':
    main()
