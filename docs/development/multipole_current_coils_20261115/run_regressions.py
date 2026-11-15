"""Run the bounded geometry, electrical, input and field-routing regressions."""
import os
from pathlib import Path
import subprocess
import sys


ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
TESTS=(
    'test_multipole_winding.py',
    'test_multipole_coil_field.py',
    'test_multipole_winding_inputs.py',
    'test_multipole_current_cache.py',
    'test_cad_multipoles.py',
    'test_probe_hexapole_support.py',
    'test_cad_probe_winding.py',
    'test_transverse_coil_field.py',
    'test_transverse_coil_wave_policy.py',
    'test_potential_capture.py',
    'test_instrument_magnetic.py',
    'test_unified_field_runtime.py',
    'test_posed_column_fields.py',
    'test_potential_capabilities.py',
    'test_component_persistence.py',
    'test_probe_winding_inputs.py',
    'test_transverse_winding_inputs.py',
    'test_winding_parameter_information.py',
    'test_parameter_impact.py',
    'test_stigmator_hardware.py',
    'test_model_inspector.py',
    'test_transverse_winding.py',
    'test_transverse_current_cache.py',
    'test_round_lens_cache_identity.py',
    'test_round_lens_models.py',
    'test_mixed_stem_model_admission.py',
    'test_part_model_document.py',
)


def main():
    environment={**os.environ,'QT_QPA_PLATFORM':'offscreen'}
    result=subprocess.run([sys.executable,'-m','pytest','-q',
        *(str(ROOT/'tests'/name) for name in TESTS),
        f'--junitxml={HERE / "validation/final-regression-tests.xml"}'],
        cwd=ROOT,env=environment,check=False)
    raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
