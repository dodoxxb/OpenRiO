Retained experiment records
=============================

reproduction.json retains complete primary/calibrated evaluation records,
independent metric checks, annotation identity and export identity.
optimizations.json retains matched results, positive-recipe checks, online
adaptation trajectories and the scoped CUDA Graph measurements.
continuation_curves.csv contains retained continuation observations, not a
complete from-scratch training history. Settings are in provenance/.
mechanism_checks.json records report-figure values, example-selection rules
and controlled probes. Failures are discussed in the PDF; experimental
deployment code and failed weights are not included.
research_status.json is the dated evidence cut for pending X/FAIR/DOTA work.
runtime_validation.json preserves earlier installation/runtime checks.
reference_audit.csv records the report's 37 cited references, 26 verified DOIs
(14 publisher, 12 arXiv), 11 official-source entries without a confirmed or
applicable DOI, version differences and the two relevant added references.
The audit checked 39 entries and dropped two unused legacy bibliography items.
Checked 2026-10-16.
package_validation.json and verify_release.log describe this package's checks.

Small JSON receipts are consolidated by name with their recorded values
preserved. Raw prediction tensors, duplicate progress snapshots and report
build files are not distributed. This does not create new experimental runs.
