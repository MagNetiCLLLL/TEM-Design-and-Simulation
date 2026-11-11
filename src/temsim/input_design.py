"""Portable input-only designs; never authorizes reuse of historical results."""
from hashlib import sha256
from pathlib import Path, PurePosixPath, PureWindowsPath

from temsim.immutable_json import thaw_json


def _archive_path(value):
    windows_path = PureWindowsPath(value)
    return windows_path if windows_path.is_absolute() else PurePosixPath(value)


def restore_input_design(snapshot):
    """Rebase unchanged bundled inputs to this checkout and return fresh state.

    Only known text-file newline differences are permitted. Archived identities
    remain untouched, and calculated checkpoints never call this entry point.
    """
    from temsim.paths import CONFIG_ROOT
    from temsim.instrument_snapshot import decode_instrument
    from temsim.physics.illumination import illumination_config

    if "archived_inputs" in snapshot.graph:
        from temsim import input_io
        from temsim.optics.electron_gun.source_policy import require_physical_gun_source
        result = decode_instrument(snapshot.graph)
        with input_io.input_scope(result, inherit=False):
            for row in snapshot.external_inputs:
                if sha256(input_io.read_bytes(row["path"])).hexdigest() != row["sha256"]:
                    raise ValueError(f"Archived input changed: {row['role']}")
            illumination_config(result)
            result.electron_gun.validate()
            require_physical_gun_source(result.electron_gun)
        return result

    catalog = next((_archive_path(row["path"]) for row in snapshot.external_inputs
                    if row["role"] == "assembly:catalog"), None)
    old_root = (catalog.parent.parent if catalog is not None
                and catalog.name == "catalog.toml" and catalog.parent.name == "instruments" else None)
    replacements = {}
    for row in snapshot.external_inputs:
        original = _archive_path(row["path"])
        target = Path(row["path"])
        in_config_root = old_root is not None and original.is_relative_to(old_root)
        if in_config_root:
            target = CONFIG_ROOT.joinpath(*original.relative_to(old_root).parts)
        try:
            content = target.read_bytes()
        except OSError as exc:
            raise ValueError(f"Missing {row['role']}: {target}; input design cannot be applied") from exc
        if sha256(content).hexdigest() != row["sha256"]:
            archived = bytes.fromhex(row["content_hex"]) if row["content_hex"] is not None else None
            # Source-controlled text can use LF or CRLF on different hosts.
            same_text = (archived is not None and target.suffix.lower() in {".toml", ".cif"}
                         and content.replace(b"\r\n", b"\n") == archived.replace(b"\r\n", b"\n"))
            if not same_text:
                raise ValueError(f"Changed {row['role']}: {target}; input design cannot be applied")
        if in_config_root and target != original:
            replacements[row["path"]] = str(target)
            # Directory references in the object graph (notably assembly.root)
            # are portable only when backed by these verified external files.
            for parent in original.parents:
                if parent == old_root or parent.is_relative_to(old_root):
                    replacements[str(parent)] = str(CONFIG_ROOT.joinpath(*parent.relative_to(old_root).parts))

    def rebase(value):
        if isinstance(value, str):
            return replacements.get(value, value)
        if isinstance(value, dict):
            return {key: rebase(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rebase(item) for item in value]
        return value

    result = decode_instrument(rebase(thaw_json(snapshot.graph)))
    illumination_config(result)
    result.electron_gun.validate()
    return result
