Publish OpenRiO v0.1.0-preview.7
=================================

Use the source ZIP for the GitHub repository. Its README.md, openrio.py,
docs/, logs/ and report/ are directly at the archive root. Extract and upload
those contents to the repository root, so README.md is the homepage.
The Chinese homepage is README.zh-CN.md. Keep the single overview image in
docs/assets/ for both pages. report/main.pdf is the only report file.

The full ZIP contains an enclosing OpenRiO/ directory and all ten detector
weights plus two source-bound adaptation snapshots. Keep .pth files out of
Git history; publish them as GitHub Release assets or distribute the full ZIP.
The historical optimization source weights are necessary for reproducing
their matched gains. They are separate from the primary N/S/M/L references.

After extracting the full package, run::

    python tools/verify_release.py

The source ZIP requires companion weights before that identity check can
pass. Each archive's SHA256SUMS covers its own included files. Preserve the
licenses, attribution notices, models.json and weight directory layout.
The READMEs point readers at the Releases page rather than hard-coding asset
URLs, so they stay correct when new Release versions are published.
The report and README identify X, FAIR1M, DOTA and portable training as
pending work; the status record has a fixed evidence date.

This preview changes packaging and bibliography, not model or runtime bytes.
The report PDF is mirrored on ResearchGate; the canonical source of the code
and evidence records is this repository, and the weight tensors live in its
Release assets. No arXiv submission has been performed.
