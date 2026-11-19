"""Expand exact frozen JSON copies for inspection or the field-fit scripts."""
import gzip
from pathlib import Path


HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    for source in HERE.rglob("*.json.gz"):
        target = source.with_suffix("")
        assert target.resolve().is_relative_to(HERE.resolve())
        original = gzip.decompress(source.read_bytes())
        if target.is_file():
            if target.read_bytes() != original:
                raise ValueError(f"Existing generated file differs from frozen evidence: {target}")
        else:
            target.write_bytes(original)
        print(target.relative_to(HERE))
