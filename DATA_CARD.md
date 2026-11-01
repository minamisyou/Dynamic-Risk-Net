# Data card

## License and scope

Original DRN contributions in this release, including the sample risk annotations
and supplied checkpoints, are licensed for noncommercial scientific or scholarly
research under the root `LICENSE`. Redistribution and adaptations must retain
the required notices and comply with that license. Separately applicable
third-party licenses and nuScenes dataset terms remain in effect; see
`THIRD_PARTY_NOTICES.md`.

## Labels in this release

`sample200/labels.csv` contains human annotations transcribed from the supplied A1–A3 workbook. It has 1,886 lane rows representing 1,556 unique lane units. Each unit retains all three annotator ratings. A strict majority is stored as `label` with `source=majority`; the 26 units without a strict majority retain their raw ratings and have a blank label with `source=unresolved`.

The human majority counts are L0–L4: 1,014 / 248 / 152 / 61 / 55 (1,530 resolved units). The CSV preserves its lane-row layout and timing columns.

## Sample

The fixed sample has 200 keyframes, one per nuScenes scene, split 140/30/30. It covers 1,556 units in 1,886 lane rows. `sample200/scenes.csv` provides the keyframe tokens and split assignments used by the build command.

nuScenes sensor data, annotations and maps are not included.
