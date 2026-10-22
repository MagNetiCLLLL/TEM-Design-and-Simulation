# Third-party notice archive

This directory preserves third-party license, copyright, attribution and NOTICE
texts for the installed CPU dependency environment reviewed for the GPLv3
migration of TEM Design & Simulation. These texts retain their own licenses;
the application's GPL-3.0-only license does not replace them.

See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for the scope, dependency
summary and redistribution considerations.

## Contents and provenance

- `inventory.json` records the 17 required top-level dependencies and their
  installed transitive closure: 74 distributions. Each entry includes the exact
  installed version, original license metadata, applicable dependencies and
  copied notice paths with SHA-256 hashes. Principal-license summaries are not
  an exhaustive license expression for every native library or data file.
- `packages/<distribution>/` contains byte-for-byte copies from installed
  distributions. The inventory's `source_path` identifies the original file
  within that distribution. Each destination is named by a 12-character
  source-path hash followed by `.txt`, keeping Windows packaging paths short.
  The collector checks for collisions within each distribution. Only filenames
  change; original paths remain in the inventory.
- `upstream/` supplies missing or supplementary texts from versioned upstream
  projects or the license steward. URLs, target package versions, byte counts
  and SHA-256 hashes are recorded in `supplements.json` and `inventory.json`.
- `collect.py` regenerates the installed inventory using standard-library
  metadata and `packaging`. It imports no numerical or GUI package and does not
  install software or access the network. It is a maintenance tool, not a
  third-party license text.

The collector excludes Qt's commercial-license pointer and preserves the
applicable upstream open-source license texts instead. This does not select
the same license for every Qt module or supply every native Qt component's
copyright notice. A Qt binary redistribution needs the notices and source
information for the actual modules and build being shipped.

## Refreshing the archive

From the project root, with the intended environment installed:

```powershell
.venv\Scripts\python.exe third_party_licenses\collect.py
```

Python 3.11 or newer and `packaging` are needed for this maintenance command.
No optional extras are selected. Requirements are traversed using the marker
environment recorded in the resulting inventory. This records what is
installed; it is neither a lockfile nor a guarantee that a future resolver
will install the same versions.

When a supplemented package version changes, the collector stops rather than
silently applying a text obtained for a different version. Review the new
upstream release and update the supplement's text, URL, version and hash.
It deliberately does not delete old notice files: review and remove obsolete
entries when updating the archive, then check that every retained file is
accounted for. Recheck manually that bundled native libraries, fonts and data
have their relevant notices; a filename-based scan is not a full provenance
audit.

Including these texts does not by itself fulfill corresponding-source,
installation-information, relinking, modification-marking or attribution
requirements. Reassess the actual contents whenever producing a frozen
application, offline environment, container or other dependency bundle.
