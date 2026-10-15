Model cards
=============

Primary N/S/M/L are unchanged EMA. Calibrated N is a separate optimization.
All assets are single-ancestry, historically test-selected community weights.

n
---

Role: primary_reproduction. Lineage: public5d.
Historical selected AP50: 71.89084724.
Batch-one source reinference AP50/AP75: 71.89083442 / 46.05460778.
Fresh source inference and independent CPU polygon/AP checks passed.
Weight: weights/n.pth. SHA-256: 7028f89ea8256e16fc2b30052c47c61697542a01288a2519e70407ee9e1c08ec.
Config: configs/n.yml. SHA-256: ec07ffbb0b2aeadf1d7cf7b08820fdf4e4f05390792e145ef0911e1f2a9226c9.
Source: mechanism_20261008/nano_pair/control/checkpoint015.pth.
Source SHA-256: e5279132c915194745916ae129599c0f5d6b5e6422cb3e650f148e0e3b7fad4a.
State: ema.module.
Export identity and package/runtime evaluation are separately recorded under logs/.

s
---

Role: primary_reproduction. Lineage: compact4d.
Historical selected AP50: 75.01996413.
Batch-one source reinference AP50/AP75: 75.02020225 / 51.54651754.
Fresh source inference and independent CPU polygon/AP checks passed.
Weight: weights/s.pth. SHA-256: 89db236d0392a3232fc0736e02ee85017616cdc6bf73259ed2ca8dc2994efa46.
Config: configs/s.yml. SHA-256: a519eac23d89a6a2c48d2e3d4f9f66c66adc5cfab904306c74ff70e9a874cd1f.
Source: scale_20261008_lx/restart_s/lr1e6/window03/checkpoint019.pth.
Source SHA-256: 5876d3f2ee3e3c60da2677095cd9b91d9770ded8b7fb8385f581cc9b07442bdf.
State: ema.module.
Export identity and package/runtime evaluation are separately recorded under logs/.

n_calibrated
--------------

Role: post_training_optimization. Lineage: public5d.
Historical selected AP50: 71.97834373.
Weight: weights/n_calibrated.pth. SHA-256: 41760a7b67296c0fdf7acb2a9262fec0204aba0dc08689d6ce86d6caaeb8c4a2.
Config: configs/n_calibrated.yml. SHA-256: ec07ffbb0b2aeadf1d7cf7b08820fdf4e4f05390792e145ef0911e1f2a9226c9.
Source: mechanism_20261008/nano_pair/higher_lr/final_ema_calibrated.pth.
Source SHA-256: c3c2d6997a110943027d22d3c9dde73d0f8e14c137990a557cd3796d64b004a0.
State: model.
Export identity and package/runtime evaluation are separately recorded under logs/.

m
---

Role: primary_reproduction. Lineage: compact4d.
Historical selected AP50: 76.44826800.
Batch-one source reinference AP50/AP75: 76.44814759 / 54.05009307.
Fresh source inference and independent CPU polygon/AP checks passed.
Weight: weights/m.pth. SHA-256: e3394902953785d79721493dd2b552668262aac4e38220af6f979bf2f2e0a823.
Config: configs/m.yml. SHA-256: 79005cebaa60d7a6c7d7570fdfce978b186dfeb07dad0c9dccaee3a28f418c98.
Source: theory_20261011/m_continuation/window00/last.pth.
Source SHA-256: db9f71947fffcb302894847e13ee64547edf1cb1e35b9b14831a6a06f81e13d0.
State: ema.module.
Export identity and package/runtime evaluation are separately recorded under logs/.

l
---

Role: primary_reproduction. Lineage: compact4d.
Historical selected AP50: 76.19903833.
Batch-one source reinference AP50/AP75: 76.19856685 / 53.81474625.
Fresh source inference and independent CPU polygon/AP checks passed.
Weight: weights/l.pth. SHA-256: c41a078f0b0ab6cd9e6fd978098bf6475ba425add196447d31390f48859f24fa.
Config: configs/l.yml. SHA-256: 8515875653396e108ddeec742482515c3fc8ac8204a8159e4d54bfc571bb2cb3.
Source: theory_20261011/l_continuation/window02/checkpoint015.pth.
Source SHA-256: 31e737e2742904838e5a4914eb5c15cbddaeb5c4e0326398be2a11919378939b.
State: ema.module.
Export identity and package/runtime evaluation are separately recorded under logs/.

Calibration N belongs to its own 71.8030 parent, not primary N.
Training-calibration subset seeds are not independent detector seeds.
See PROTOCOL.rst, NOTICE.txt and WEIGHTS_NOTICE.txt for limitations and terms.

Optimization source assets
----------------------------

n_compact_multiview and n_rotation_multiview bind the two Nano inference
comparisons to their original source states. s_multiview and m_multiview
are the earlier Small/Medium sources used in the full-set transfer study.
They are required for the reported optimization deltas and do not replace
the latest single-view primary models. n_calibration_parent is the exact
uncalibrated 71.8030 source of the included n_calibrated 71.9783 state.

The two files under adapters/ contain terminal teacher/student normalization
parameters from positive synthetic-noise online experiments. Their metadata
binds each snapshot to an exact source digest. Their reported stream AP does
not evaluate the terminal state as a static detector. See
OPTIMIZATION_GUIDE.rst for applicable commands and controls.
