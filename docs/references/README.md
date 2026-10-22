# Reference sources and local reading copies

This directory keeps the project's authored research notes, source links and
provenance metadata. Third-party PDFs, downloaded HTML, slide decks, page
renders and screenshots are local reading copies: they are excluded from Git
tracking and source distributions. They are not licensed by the project's
root license. Public availability and attribution alone do not establish
permission to redistribute an original document.

Existing local copies stay at their original paths so that local notes and
hash checks remain useful. On a fresh clone, use the original source URLs in
the notes or `source.json` files; links to local copies will not resolve until
the reader separately obtains those files with appropriate permission. An
`original_pdf_saved` or similar status records a past local acquisition, not
a promise that the file is supplied by this repository.

Put other private reference material in `docs/references/local-only/`, which
is ignored as a whole. Do not force-add reading copies or attach them to
releases. Simulator-generated illustrations and runtime assets elsewhere in
the project are outside this exclusion.

Removing a file from current tracking does not remove it from earlier Git
commits or remote copies. Cleaning previously uploaded history requires a
separate, coordinated operation. Before another checkout pulls a commit that
removes previously tracked reference files, back up that checkout's reading
copies outside the repository; pulling the removal can delete those copies.
