"""Exact coherent line readouts, not a two-dimensional beam distribution."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.read_gaussian_phase_plane import load_packets, sha_file
import argparse
import hashlib
import json
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("packets", type=Path)
parser.add_argument("--width-nm", type=float, default=40.)
parser.add_argument("--pixels", type=int, default=5001)
parser.add_argument("--tag", default="profiles")
args = parser.parse_args()
if args.pixels % 2 != 1:
    raise ValueError("Central line profile requires odd pixels")
receipt_path = args.packets.with_name(args.packets.name.replace("-packets.npz", "-receipt.json"))
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
receipt_sha = sha_file(receipt_path)
packet_sha = sha_file(args.packets)
p = load_packets(args.packets)
dx = args.width_nm*1e-9/(args.pixels-1)
axis = (np.arange(args.pixels)-args.pixels//2)*dx
fields, records = {}, {}
started = time.perf_counter()
for name, basis in (("x", np.eye(2)*dx), ("y", np.array([[0., -dx], [dx, 0.]]))):
    field, record = p.sample_plane((1, args.pixels), basis, np.zeros(2), method="recurrence",
                                   return_record=True, maximum_working_bytes=256*1024**2)
    fields[name] = field[0]/dx  # cell amplitude -> physical amplitude density
    records[name] = record
if sha_file(args.packets) != packet_sha or sha_file(receipt_path) != receipt_sha:
    raise ValueError("Packet or replay receipt changed during the line readout")
out = args.packets.with_name(args.packets.stem.replace("-packets", "")+"-"+args.tag)
np.savez(out.with_suffix(".npz"), axis_m=axis, x_amplitude_density=fields["x"], y_amplitude_density=fields["y"])
record = {"scope": "Physical coherent complex field on two laboratory axes through X=Y=0; not integrated marginal profiles or a two-dimensional whole-beam image",
    "source_receipt": str(receipt_path), "source_receipt_sha256": receipt_sha,
    "source_packet": str(args.packets), "source_packet_sha256": packet_sha,
    "mode_id": receipt["mode_id"], "energy_kev": receipt["energy_kev"], "mode_weight": receipt["weight"],
    "sample_entrance_z_mm": receipt["stop_z_mm"], "pitch_nm": dx*1e9, "width_nm": args.width_nm,
    "pixels": args.pixels, "normalization_fitted": False, "piston_removed": False,
    "evaluation": records, "elapsed_s": time.perf_counter()-started}
out.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps({"output": str(out), "elapsed_s": record["elapsed_s"]}), flush=True)
