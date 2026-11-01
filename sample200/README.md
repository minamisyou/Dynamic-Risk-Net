# sample200 — human annotations

The original DRN annotation contributions are provided for noncommercial
scientific or scholarly research under `../LICENSE`. Third-party materials retain
their own terms; see `../THIRD_PARTY_NOTICES.md`.

This fixed sample contains 200 nuScenes keyframes (one per scene; train/val/test 140/30/30). It has 1,556 unique lane units expanded to 1,886 lane rows. `labels.csv` contains the A1–A3 human ratings and strict-majority outcomes.

## Human annotation counts

| measure | count |
|---|---:|
| unique units | 1,556 |
| strict majority, L0–L4 | 1,014 / 248 / 152 / 61 / 55 |
| no strict majority | 26 |
| lane rows | 1,886 |

The three raw ratings are retained on every lane row. Majority labels are copied to every lane row for the corresponding unit. Units without a strict majority have blank `label`, `source=unresolved`, and are ignored by model loss. Timing and split columns remain in the existing lane-row format.

## Files

- `labels.csv`: lane-row annotations, raw A1–A3 ratings, consensus labels, timing fields and split.
- `scenes.csv`: sample keyframe tokens, scene names, split assignments and scenario flags.
- `scenes.jsonl`: scene-level sample records.
- `splits.json`: scene-to-split mapping.
