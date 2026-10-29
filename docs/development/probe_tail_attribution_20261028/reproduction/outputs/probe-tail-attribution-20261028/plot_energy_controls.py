"""Plot actual energy controls without fitting, refocusing or current scaling."""
from pathlib import Path
from hashlib import sha256
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
source = HERE / "energy-comparison-poly-constant-mono/comparison.json"
records = json.loads(source.read_text())["variants"]
cases = [("poly_baseline", "Original energy spread", "#155b8a"),
         ("poly_phase_changed_energy", "Same gun-exit phase; constant column energy", "#d77916"),
         ("full_mono_gun_and_column", "Monochromatic emission and full gun", "#16805a")]
fig, axes = plt.subplots(1,2,figsize=(13.6,5.5),constrained_layout=True)
fig.suptitle("Energy controls with fixed optical excitations",fontsize=17)
for key,label,color in cases:
    entry = records[key]
    data_path = Path(entry["path"]) / "particles.npz"
    assert sha256(data_path.read_bytes()).hexdigest() == entry["particles_sha256"]
    rows = entry["planes"]
    alpha = entry["surface"]["alpha95_mrad"]
    axes[0].plot([r["offset_nm"] for r in rows],[r["d95_nm"] for r in rows],
        color=color,lw=1.9,label=f"{label}\nalpha95 = {alpha:.3f} mrad")
    minimum = entry["d95_minimum"]
    axes[0].plot(minimum["representative_offset_nm"],minimum["minimum"],"o",color=color,ms=5)
    with np.load(data_path,allow_pickle=False) as data:
        i = int(np.flatnonzero(data["z_mm"] == entry["surface"]["z_mm"])[0])
        mask = data["alive"][i] & (data["weights"] > 0)
        weights = data["weights"]
        x,y = data["x_m"][i,mask],data["y_m"][i,mask]
        w = weights[mask]
        center = (np.average(x,weights=w),np.average(y,weights=w))
        diameter = 2e9*np.hypot(x-center[0],y-center[1])
        order = np.argsort(diameter)
        cumulative = np.cumsum(w[order])/weights.sum()
        axes[1].step(diameter[order],cumulative*100,where="post",color=color,lw=1.8)
axes[0].axvline(0,color="black",ls="--",lw=1)
axes[0].set(xlim=(-40,45),ylim=(1.4,4.2),xlabel="Distance from specimen upper surface (nm)",
            ylabel="Native D95 (nm)",title="Executed through-focus observations")
axes[0].legend(fontsize=8.7,loc="upper right",framealpha=.95)
axes[1].set(xscale="log",xlim=(.15,75),ylim=(50,100.4),xlabel="Centered containment diameter (nm)",
            ylabel="Fraction of original current (%)",title="Actual specimen-plane current distribution")
for value in (90,95):
    axes[1].axhline(value,color="gray",ls=":",lw=.9)
for ax in axes:
    ax.grid(True,alpha=.22)
fig.supxlabel("All cases retain 769 electrons and 10 uA. Full monochromatic emission also changes the gun-exit phase and convergence angle.",fontsize=10)
for suffix in ("png","svg"):
    fig.savefig(HERE / ("energy-controls."+suffix),dpi=180)
(HERE / "energy-controls-plot-provenance.json").write_text(json.dumps(dict(
    source=str(source),source_sha256=sha256(source.read_bytes()).hexdigest(),
    script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
    interpretation="Lines connect executed observations; no interpolated waist certificate or size acceptance cap"),indent=2)+"\n")
