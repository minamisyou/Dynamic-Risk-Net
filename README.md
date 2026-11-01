# Dynamic-Risk Net — DRN code, checkpoints and 200-keyframe sample

This bundle contains a DRN implementation, released checkpoints, a fixed 200-keyframe nuScenes sample and the human A1–A3 annotations for that sample. Read `DATA_CARD.md` before interpreting the labels.

## License

Original DRN code, annotations, model checkpoints, and documentation are available
under the **Dynamic-Risk Net Noncommercial Research License 1.0** in [`LICENSE`](LICENSE).
Noncommercial scientific or scholarly research use, modification, and
redistribution are permitted under its conditions. Commercial use, commercial
research and development, and production deployment require separate permission.
This is a research release under a custom license; it is not licensed under MIT.
Third-party materials retain their own terms; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

| folder | content |
|---|---|
| `sample200/` | 200 keyframes (train 140 / val 30 / test 30); `labels.csv` contains transcribed human ratings and strict-majority labels |
| `code/` | human review/annotation CLI (`drn`), model/train/evaluation CLI (`drn_model`) and tests |
| `checkpoints/` | DRN seeds 0–2, model weights and training configurations |

## Quick start

Use Python 3.10.9 and install the pinned packages in `code/requirements.txt`. Set `dataroot` in `code/config.yaml` to a nuScenes v1.0-trainval directory with map expansion v1.3. nuScenes sensor data and maps are not included.

From `code/`, build the released sample, prepare the human labels, train and evaluate:

```sh
python -m drn build --only-tokens ../sample200/scenes.csv
python -m drn_model prepare --out ~/data/drn_dataset --labels ../sample200/labels.csv --cache ~/data/drn_dataset/sample200.pt
python -m drn_model train --cache ~/data/drn_dataset/sample200.pt --run ~/data/drn_runs/drn_s0 --seed 0
python -m drn_model evaluate --cache ~/data/drn_dataset/sample200.pt --run ~/data/drn_runs/drn_s0 --split test
```

Unresolved units are ignored by the loss while their raw A1–A3 ratings remain available in the CSV. The `drn` command also provides `review`, `annotate`, `consensus`, `export`, `sheets` workflows.
