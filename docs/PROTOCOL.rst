DIOR-R evaluation and training disclosure
===========================================

The dataset directory is resolved through the command's --data-root argument::

    DIOR-R/
      JPEGImages-test/
      JPEGImages-trainval/
      ImageSets/Main/train.txt
      ImageSets/Main/val.txt
      ImageSets/Main/test.txt
      Annotations/Oriented Bounding Boxes/

Evaluation uses 11,738 test images, 800-pixel square input, the configured final
decoder output and 300 retained detections. Report AP50 and AP75 as percentages
using the included rotated polygon evaluator and VOC2007 eleven-point AP.
Classes, difficult-target treatment, angle conversion and matching are fixed
by the shipped configuration and evaluator, whose hashes are recorded. These
are not COCO AP50:95 or official DOTA server scores.

The DIOR XML parser follows the included upstream implementation: it converts
polygons using OpenCV minimum-area rectangles, orders the long side first and
stores the angle divided by 180 degrees in [0, 1). Postprocessing/evaluation
uses radians. The parser does not read the XML difficult flag and does not emit
ignored boxes; valid parsed objects are evaluated. The full test-split flag
inventory is in ``logs/reproduction.json``. This behavior must be
preserved when comparing the released metrics; alternate difficulty rules
require separately labeled evaluation.
The inspected test XMLs contain 124,445 objects, all with difficult=0, so this
parser distinction does not change difficult-object inclusion on that split.

Zero-based class order is airplane, airport, baseballfield, basketballcourt,
bridge, chimney, expressway-service-area, expressway-toll-station, dam,
golffield, groundtrackfield, harbor, overpass, ship, stadium, storagetank,
tenniscourt, trainstation, vehicle, windmill.

Original RiO-DETR credit is retained. The primary Nano reconstructs the public
five-coordinate geometry path. Small, Medium and Large use the explicitly disclosed compact
four-coordinate geometry path with zero denoising reference-angle input.
Final-stage recipes use frozen normalization, quarter-turn augmentation and
matched low-learning-rate continuation. State, optimizer resets, EMA averaging
and batch size matter; changing only the YAML is not enough to reconstruct
every historical stage. Historical specs and source hashes are supplied as
provenance. A portable from-scratch multi-stage training command is pending.

Our main table uses uncalibrated EMA. Raw weights, normalization calibration,
multi-view predictions, model ensembles and test-time parameter adaptation
are separate interventions. The calibrated Nano uses training-side images
and changes two normalization layers' running moments. It has a different
parent from the primary Nano. Subset checks are not detector-seed replication.

All AP in the primary table uses the full standard test split. Repeated test
feedback guided checkpoint/recipe selection, and 23 exact cross-split images
are documented. Therefore these are exploratory, single-ancestry community
results; they do not establish unbiased generalization or explain remaining
gaps by hardware noise. Dataset terms and class definitions remain upstream.

FAIR1M has both native patch and original-scene merged evaluation. These are
not interchangeable; paper-protocol equivalence remains open. Local DOTA
exports are not official server measurements.

Independent metric check
--------------------------

After complete native evaluation, recompute AP from its saved predictions::

    python tools/independent_eval.py --data-root /path/to/DIOR-R \
      --predictions outputs/l-full/predictions.pt \
      --native-metrics outputs/l-full/metrics.json \
      --output outputs/l-independent

This CPU-only tool independently parses test XML and uses OpenCV polygon
intersection and greedy VOC07 matching. It checks all 11,738 image IDs and
records the annotation digest. Only load prediction files you trust. The
0.01-point agreement gate checks numerical implementation, not paper-protocol
equivalence, dataset independence or model-seed robustness. The preview's
default evaluation batch is one, matching the latest source audits; changing
batch size may change floating-point boundary decisions.
