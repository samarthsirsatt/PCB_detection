# PCB Imaging and Localization Study

## 1. Objective

The purpose of this study was to examine whether the variation in image field of view (FOV) in the existing PCB dataset has a measurable effect on defect localization performance. The study also evaluates whether the number of pixels representing a defect is more closely related to localization performance and considers the implications for future camera and lens selection.

## 2. Existing Dataset and Imaging Characteristics

The analysis was performed using the existing PCB image dataset and its COCO annotations; no additional images were collected for this study.

The main imaging characteristics are:

| Parameter | Observed value |
|---|---:|
| Image resolution | 1280 × 720 px |
| Horizontal FOV | 1.2–40 mm |
| Median FOV | ~18 mm |
| Sampling density | 32–1067 px/mm |
| Median sampling density | ~71 px/mm |
| Small frequent defect | Solder Ball |
| Median annotated Solder Ball size | ~0.77 mm |

The calculated px/mm values represent image sampling density. They should not be interpreted as the actual optical resolution of the imaging system, since camera and lens optical characteristics are not available in the current metadata.

## 3. FOV and Localization Analysis

The existing RT-DETR model with 30 object queries, which was the best-performing configuration in the current experiments, was evaluated on the original validation split without retraining.

The evaluation included 125 validation images containing 107 annotated defect instances. The model reproduced the previously recorded performance, confirming that the evaluation procedure was consistent.

The relationship between FOV-related sampling density and localization quality was then examined using the best IoU obtained for each defect instance.

The result was:

- Spearman correlation between px/mm and localization IoU: **ρ = −0.028**
- Permutation-test significance: **p ≈ 0.69**

This indicates that no statistically significant relationship between FOV-related sampling density and localization IoU was observed in the current experiment.

However, defect pixel size showed a more noticeable relationship with localization quality:

- Spearman correlation between defect pixel size and localization IoU: **ρ = +0.294**
- Statistical significance: **p ≈ 0.002**

Therefore, within the limits of the current dataset and experiment, the number of pixels representing the defect appears to be more relevant to localization performance than FOV itself.

## 4. Implications for Imaging System Design

The current analysis does not show a statistically significant relationship between FOV-related sampling density and localization IoU. Therefore, the imaging system should not be designed around megapixel count alone. Instead, the camera configuration should be derived from the required defect size, pixels per defect, and inspection area.

For the present inspection requirement, a minimum defect size of approximately **0.1 mm** has been considered. A working target of **100 px/mm** provides approximately **10 pixels across a 0.1 mm defect**:

**100 px/mm × 0.1 mm = 10 pixels**

Based on this requirement, the current camera concept is a **~25 MP camera providing approximately 5000 × 5000 pixels**. A tiled acquisition strategy can then be used for larger PCB regions.

For a **50 × 50 mm FOV per tile** at 100 px/mm:

**50 mm × 100 px/mm = 5000 pixels**

Thus, one tile corresponds approximately to **5000 × 5000 pixels (25 MP)**.

For a larger inspection region of approximately **150 × 150 mm**, a **3 × 3 tile arrangement** would provide nine 50 × 50 mm inspection regions. This provides a practical way to maintain the required sampling density while covering a larger PCB area.

This calculation provides a preliminary basis for camera selection:

| Parameter | Current design target |
|---|---:|
| Minimum defect size | ~0.1 mm |
| Target sampling density | ~100 px/mm |
| Pixels across 0.1 mm defect | ~10 px |
| FOV per tile | 50 × 50 mm |
| Required image size per tile | ~5000 × 5000 px |
| Approximate camera resolution | ~25 MP |
| Example larger coverage | 3 × 3 tiles |
| Total covered area | ~150 × 150 mm |

The 25 MP / 5000 × 5000 specification should be treated as a **preliminary system design target**, to be confirmed against the actual sensor format, pixel pitch, lens magnification, working distance, optical resolution, and field-of-view requirements.

## 5. Parameters to Confirm for the Final Hardware Configuration

The main system-level imaging targets have already been established: approximately **0.1 mm minimum defect size**, **100 px/mm target sampling density**, and approximately **50 × 50 mm FOV per tile**, corresponding to about **5000 × 5000 pixels (~25 MP)** per tile.

The remaining task is to translate these targets into an actual camera–lens configuration. The following physical parameters should therefore be confirmed:

1. Actual camera sensor dimensions and pixel pitch.
2. Lens/objective magnification and working distance.
3. Achievable 50 × 50 mm FOV with the selected sensor and lens.
4. Optical resolution/MTF at the required working distance.
5. Distortion and image quality across the tile.
6. Lighting configuration and illumination uniformity.
7. Practical overlap and alignment between tiles in the proposed 3 × 3 acquisition strategy.

The camera should therefore be selected from the established inspection requirement and tile geometry, rather than from megapixel count alone.

## 6. Recommended Next Steps

The current results suggest that a new imaging dataset is not immediately necessary. The next stage should focus on connecting the established imaging requirements to the practical hardware configuration.

For future image collection, capture metadata should also include the camera, lens, working distance, FOV, board identity, and lighting conditions so that imaging variables can be analysed separately from model performance.

## 7. Conclusion

The imaging analysis was conducted to determine whether FOV variation in the existing PCB dataset is a major factor affecting defect localization. The current results do not show a statistically significant relationship between FOV-related sampling density and localization IoU, while defect pixel size shows a moderate positive relationship with localization quality.

The imaging requirement has already been translated into a preliminary hardware concept. For a minimum defect size of approximately **0.1 mm**, a target of **100 px/mm** provides approximately 10 pixels across the defect. This leads to a **50 × 50 mm FOV per tile requiring approximately 5000 × 5000 pixels**, corresponding to a **~25 MP camera**. A **3 × 3 tiled acquisition** can then provide approximately **150 × 150 mm total coverage** while maintaining the target sampling density.

The next stage is to verify this preliminary specification using the actual camera sensor, lens, working distance, optical resolution, and lighting configuration. The proposed values should therefore be treated as established system-design targets for hardware evaluation, with the final camera and lens selected after physical verification.
