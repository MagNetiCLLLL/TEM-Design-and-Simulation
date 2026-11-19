"""Run the bounded source-free CAD, input, field and admission regression set."""
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
TESTS = (
    'test_transverse_winding.py',
    'test_transverse_winding_inputs.py',
    'test_transverse_coil_field.py',
    'test_transverse_current_cache.py',
    'test_transverse_coil_wave_policy.py',
    'test_cad_multipoles.py',
    'test_condenser_stigmator_support.py',
    'test_gun_stigmator_geometry.py',
    'test_deflector_support_geometry.py',
    'test_ordinary_deflector_structure.py',
    'test_deflector_winding_length_contract.py',
    'test_probe_lower_deflector_details.py',
    'test_cad_probe_winding.py',
    'test_cad_round_winding.py',
    'test_component_persistence.py',
    'test_round_lens_cache_identity.py',
    'test_part_model_document.py',
    'test_parameter_impact.py',
    'test_winding_parameter_information.py',
    'test_coil_winding_dialog.py',
    'test_probe_winding_inputs.py',
    'test_potential_capture.py',
    'test_instrument_magnetic.py',
    'test_unified_field_runtime.py',
    'test_round_lens_models.py',
    'test_potential_capabilities.py',
    'test_mixed_stem_model_admission.py',
)


if __name__ == '__main__':
    environment = {**os.environ, 'QT_QPA_PLATFORM': 'offscreen'}
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q',
        *(str(ROOT / 'tests' / name) for name in TESTS),
        f'--junitxml={HERE / "validation/final-regression-tests.xml"}'],
        cwd=ROOT, env=environment, check=False)
    raise SystemExit(result.returncode)
