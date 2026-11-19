"""Offline wheel/import/CIF checks in a detached directory, with no transport."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARTIFACTS = ROOT / "outputs/ideal-coil-si110-scan-20261116/packaging-final"


def run(arguments, **kwargs):
    return subprocess.run([sys.executable, *arguments], check=True, **kwargs)


if __name__ == "__main__":
    wheels = ARTIFACTS / "wheels"
    target = ARTIFACTS / "installed"
    elsewhere = ARTIFACTS / "elsewhere"
    wheels.mkdir(parents=True, exist_ok=True)
    elsewhere.mkdir(parents=True, exist_ok=True)
    run(["-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--no-index",
         "--wheel-dir", str(wheels), str(ROOT)], cwd=ROOT)
    wheel, = wheels.glob("*.whl")
    run(["-m", "pip", "install", "--no-deps", "--no-index", "--target", str(target), str(wheel)], cwd=elsewhere)
    environment = dict(os.environ)
    environment.pop("TEMSIM_PROJECT_ROOT", None)
    child = r'''
import json, sys
from hashlib import sha256
from pathlib import Path
target = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))
# pip --target places data files in target/configs. Bind the normal installed
# prefix explicitly for this host-interpreter check; dependencies remain in
# the project virtual environment, so this is not a fresh dependency install.
sys.prefix = str(target)
import temsim
import temsim.ideal_coil_segments
import temsim.ideal_transverse_segments
import temsim.physics.cuda_coil_potential
import temsim.physics.cuda_gun_step
from temsim.optics.column import default_state
from temsim.paths import BUNDLED_SI_CIF, CONFIG_ROOT
from temsim.specimen.cif_io import read_cif_atoms
from temsim.optics.electron_gun.emitter import ColdFieldEmitter
def forbidden(*args, **kwargs):
    raise AssertionError("Installed-input check must not emit a physical source")
ColdFieldEmitter.emit = forbidden
assert Path(temsim.__file__).resolve().is_relative_to(target)
assert CONFIG_ROOT.resolve().is_relative_to(target)
state = default_state()
assert state.lenses
atoms = read_cif_atoms(BUNDLED_SI_CIF)
assert len(atoms) > 0 and set(atoms.get_chemical_symbols()) == {"Si"}
print(json.dumps(dict(scope="Installed-target imports with explicit prefix and host dependencies; no source or wave",
    package_path=str(Path(temsim.__file__).resolve()), config_root=str(CONFIG_ROOT.resolve()),
    cif_path=str(BUNDLED_SI_CIF.resolve()), cif_sha256=sha256(BUNDLED_SI_CIF.read_bytes()).hexdigest(),
    cif_atom_count=len(atoms), installed_lens_count=len(state.lenses)), indent=2))
'''
    result = run(["-c", child, str(target)], cwd=elsewhere, env=environment,
                 text=True, capture_output=True)
    receipt = json.loads(result.stdout)
    receipt.update(wheel_name=wheel.name, wheel_sha256=sha256(wheel.read_bytes()).hexdigest())
    validation = HERE / "validation"
    validation.mkdir(exist_ok=True)
    (validation / "packaging.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
