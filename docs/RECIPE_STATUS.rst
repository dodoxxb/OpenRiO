Training recipe scope
=======================

These are recorded terminal-stage settings, not a portable full training recipe.
Full lineage decisions and retained observations are under logs/continuation_curves.csv and provenance/.

N
---
Batch 8; accumulation 1; LR 2.5e-07; backbone multiplier 0.1; frozen BN True; quarter-turn True; seed 1; resume training state False.

S
---
Batch 8; accumulation 1; LR 1e-06; backbone multiplier 0.1; frozen BN True; quarter-turn True; seed 0; resume training state True.

M
---
Batch 4; accumulation 1; LR 1e-06; backbone multiplier 0.1; frozen BN True; quarter-turn True; seed 0; resume training state True.

L
---
Batch 4; accumulation 1; LR 1e-06; backbone multiplier 0.1; frozen BN True; quarter-turn True; seed 0; resume training state True.

These settings refer to selected terminal branches and do not replace the base
training schedules. Effective batches formed by accumulated microbatch losses
are not asserted equivalent to joint large-batch loss normalization or SyncBN.
Independent seeds and a portable stage-transition runner are next-release work.
