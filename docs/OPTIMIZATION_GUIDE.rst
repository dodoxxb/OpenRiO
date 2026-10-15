Using the released accuracy and execution improvements
========================================================

This package includes useful interventions and the controls needed to measure
their benefit. Unsuccessful method implementations and experimental weights
are not shipped as deployment recipes. Their reasoning, outcomes and limits
remain in the technical report.

Choose the correct model and result
-------------------------------------

The primary N/S/M/L weights are the latest reproduction references. The four
``*_multiview`` assets are the exact earlier sources used for the published
multi-view comparisons. Both sets are useful: use the primary models for the
strongest released single-view baseline, and the named historical sources to
reproduce the stated inference deltas. Gains on the latter are not silently
added to the former. All methods below use square 800x800 input tiles.

.. list-table:: Complete DIOR-R test results; AP points
   :header-rows: 1

   * - Source
     - Comparison
     - AP50
     - AP75
   * - Compact Nano
     - Five-view NMS / weighted coordinates
     - 72.903 / 73.597
     - 45.516 / 48.575
   * - Rotation Nano
     - Five-view NMS / weighted coordinates
     - 74.247 / 74.848
     - 46.996 / 49.501
   * - Small
     - Single view / weighted coordinates
     - 73.982 / 76.465
     - 50.241 / 53.213
   * - Medium
     - Single view / uniform coordinates
     - 74.790 / 77.512
     - 52.499 / 55.430

The comparisons use their own frozen sources. The first two isolate additional
processing of a five-view pool; the last two include the benefit and cost of
five forwards. Uniform coordinates instead give 48.818/49.716 AP75 on the two
Nano paths, and 53.317 on Small. This trades AP50 against AP75; no single
choice dominates every metric. Exact values and source digests are in
``logs/optimizations.json``.

Multi-view inference without parameter updates
------------------------------------------------

The package provides ordinary five-view NMS, fixed-box rescoring, agreement
selection, and uniform or confidence-weighted geometric representatives.
The first and third are also necessary controls for attributing the benefit.
The views are original, horizontal flip and 90/180/270 degree rotations.
All inverse mappings use continuous 800-pixel edges. No annotations enter
view matching, ranking or representative selection.

Predict an image using the Small source and weighted geometry::

    python optimize.py predict --model s_multiview --method weighted \
      --image /path/to/800-square.jpg --device cuda:0 --output outputs/s-example

Evaluate all released choices on the same complete test split::

    python optimize.py evaluate --model m_multiview --method all \
      --data-root /path/to/DIOR-R --device cuda:0 --memory-fraction 0.12 \
      --output outputs/m-multiview

Each method receives its own predictions.pt and metrics.json. Independently
verify one result with the same portable polygon metric::

    python tools/independent_eval.py --data-root /path/to/DIOR-R \
      --predictions outputs/m-multiview/uniform/predictions.pt \
      --native-metrics outputs/m-multiview/uniform/metrics.json \
      --output outputs/m-multiview-independent

Other source names are ``n_compact_multiview``, ``n_rotation_multiview`` and
``s_multiview``. ``--save-views`` retains reusable five-view caches and needs
several GB of disk for a full split. ``--limit`` is diagnostic only and
suppresses AP. The same CLI accepts current primary models, but the table
above does not report those unmeasured combinations.

Training-image normalization calibration
------------------------------------------

The package includes the exact 71.8030 parent as ``n_calibration_parent`` and
the resulting 71.9783 state as ``n_calibrated``. The procedure changes only
running means/variances of two specific normalization layers, sequentially,
using 256 recorded training-side image identifiers. Learned parameters and
all other buffers remain unchanged. Reproduce the state correction::

    python calibrate.py --data-root /path/to/DIOR-R --device cuda:0 \
      --output outputs/n-calibration --verify-reference

Evaluate the already included improved weight::

    python openrio.py evaluate --model n_calibrated \
      --data-root /path/to/DIOR-R --output outputs/n-calibrated-eval

The identifier list is supplied in ``recipes/n_calibration_images.txt``.
No annotation content is read by calibration. Training/test identifiers are
checked for overlap, while the project's known byte-duplicate exposure and
adaptive test selection remain disclosed. This is a demonstrated correction
of its own parent, not a universally beneficial calibration or a correction
of the primary 71.891 Nano checkpoint.

Online test-time adaptation under synthetic noise
---------------------------------------------------

The N/S recipes adapt normalization affine parameters using native final-layer
pseudo-label losses, a dynamic teacher EMA and source anchoring. Both models
stay in evaluation mode, so normalization buffers remain frozen. Predictions
are emitted before updating from the current image. No ground-truth file is
read by adaptation; metrics are computed afterwards in a separate CPU process.
The existing protocol uses a fixed shuffled full test stream and deterministic
Gaussian image noise with sigma 0.08. The positive complete-stream references
are N +0.688 AP50 / +0.182 AP75 and S +0.743 / +0.380 against their matched
frozen noisy-stream sources. Clean streams declined in the research controls.

Run the adapted and frozen control streams in fresh directories::

    python adapt.py --recipe n_noise --data-root /path/to/DIOR-R \
      --device cuda:0 --output outputs/n-noise-adapted
    python adapt.py --recipe n_noise --frozen --data-root /path/to/DIOR-R \
      --device cuda:0 --output outputs/n-noise-frozen

Replace ``n_noise`` with ``s_noise`` for Small. The default uses the joint
classification and geometry losses. ``--loss-mode classification`` exposes
an additional positive tested loss choice; its exact gains differ. These
are parameter updates, not training-free inference, and are not recommended
as a general clean-domain accuracy switch. The dynamic EMA and anchoring rules draw on
`DESA-TTA <https://arxiv.org/abs/2609.01795>`_, with OpenRiO-specific
pseudo-label and detector-loss handling.

Score only after the complete stream finishes::

    python score_adaptation.py --run outputs/n-noise-adapted \
      --data-root /path/to/DIOR-R
    python score_adaptation.py --run outputs/n-noise-frozen \
      --data-root /path/to/DIOR-R

The scorer checks completeness and image order and produces predictions.pt
and metrics.json, which also work with tools/independent_eval.py. Adaptation
and scoring must be separate processes. ``--limit 16`` exercises real
updates and a dense 300-pseudo-target backward probe but cannot receive a
complete-stream score. CUDA execution with 800x800 images is the validated
adaptation path. The process applies an allocator cap and an admission check;
neither reserves VRAM against other processes.

``adapters/n_noise_terminal_norms.pth`` and its Small counterpart preserve
the exact teacher/student normalization parameters at the end of the original
positive runs. They are tied to the corresponding source digest and stream.
They are inspectable tensor snapshots, not standalone detectors or resumable
optimizer checkpoints. The reported AP belongs to the online trajectory,
not to static inference with these terminal parameters. Reproduction starts
from weights/n.pth or weights/s.pth and reruns the supplied recipe. Numerical
and ordering differences after backward updates are recorded; no bitwise
cross-run trajectory guarantee is claimed.

Output-preserving CUDA Graph execution
----------------------------------------

Benchmark the primary Nano source with paired execution and output checks::

    python benchmark.py --model n --data-root /path/to/DIOR-R \
      --device cuda:0 --output outputs/n-graph

The measurement starts with compressed image bytes already in host memory,
then includes decode, preprocessing, transfer, forward, top-300 postprocessing
and owned CPU outputs. Disk/network/serialization and graph capture are
excluded from the paired per-image latency; capture time is reported separately.
New timing results depend on the local workload. The earlier shared-device
34.17 to 30.36 ms result is a contextual reference, not a promised speedup.
The graph object reuses output buffers, so a caller retaining results must
copy them before the next replay. This path retains all decoder queries;
reduced-query variants with accuracy loss are not shipped as recommended tools.

Verification and remaining scope
----------------------------------

The exported historical weights have exact tensor identity checks. Four
sources receive fresh spread-index five-view inference checks and complete
cached-prediction comparisons of the core inference methods. CPU-only clean
environment loading checks cover all ten detector assets. Calibration is
checked against its included reference. Adaptation's loss helpers, operators
and update loop are checked against the original implementation; fresh
complete-stream re-evaluation has now passed for both N/S. Fresh online
AP50/AP75 are 57.798739/36.232103 and 57.226896/39.109911; deltas to
the retained matched frozen controls are +0.726654/+0.203875 and
+0.774967/+0.392308. The frozen controls were not rerun in this addendum.
These same-source reruns do not estimate independent-training variability.
See ``logs/optimizations.json``.

The preview does not package the separate YOLO quality-head experiment as a
supported model integration; its positive transfer evidence and controls
remain in the paper. End-to-end training recipes and independent detector
seeds remain separate roadmap items. No claim of a new general algorithm or
universal domain-adaptation gain follows from this selected deployment set.
