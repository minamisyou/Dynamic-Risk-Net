# Third party materials

The root `LICENSE` applies to original Dynamic-Risk Net contributions. Existing
third-party licenses and dataset terms remain applicable to third-party material
and are not replaced by the DRN license.

## nuScenes

The sample uses nuScenes scene, sample, and lane identifiers and human risk
annotations associated with those scenes. The risk ratings and consensus outcomes
are the DRN annotations; they are not official nuScenes risk labels.

nuScenes sensor data, original annotations, and maps are not included. Users must
obtain the required nuScenes data and map expansion separately, and comply with
the terms applicable to those materials and their use:

- Dataset website: https://www.nuscenes.org/
- Dataset terms: https://www.nuscenes.org/terms-of-use

The DRN license does not grant rights on behalf of nuScenes or its rights holders.
Any independently applicable upstream license remains in effect.

Dataset reference: Holger Caesar et al., "nuScenes: A multimodal dataset for
autonomous driving," CVPR, 2020.

## Dependencies

The packages listed in `code/requirements.txt`, including PyTorch and the nuScenes
devkit, are third-party software and are not relicensed by this release. Obtain
them from their respective distributions and retain their applicable notices.
The root license does not change the licenses of those dependencies.
