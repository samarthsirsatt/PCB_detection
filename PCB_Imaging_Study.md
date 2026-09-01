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

The results do not currently support the conclusion that variation in FOV is a primary cause of poor localization performance. Therefore, increasing camera resolution solely to compensate for FOV variation is not yet justified by the available evidence.

A more useful basis for camera selection is the required number of pixels across the smallest defect of interest:

**Required pixels/mm = desired pixels across defect ÷ defect size**

For example, based on the measured defect-size distribution:

- A Solder Ball of approximately **0.449 mm** at the lower end of the observed size distribution would require about **22 px/mm** for 10 pixels across the defect.
- The same defect would require about **45 px/mm** for 20 pixels across the defect.
- A small LED Damage instance of approximately **0.259 mm** would require about **77 px/mm** for 20 pixels across the defect.

The current dataset already covers approximately **32–1067 px/mm**, indicating that a wide range of sampling densities is already represented in the data. Consequently, there is not sufficient evidence at this stage to justify selecting a higher-megapixel camera solely on the basis of the current localization results.

## 5. Information Required Before Camera Selection

A final camera and lens specification cannot yet be determined because several physical imaging parameters are not available in the current dataset. The following should be fixed before hardware selection:

1. Target PCB/component or inspection ROI.
2. Required inspection area and corresponding FOV.
3. Smallest defect that must be detected or localized.
4. Required pixels across that defect.
5. Camera sensor resolution and pixel size.
6. Lens/objective and working distance.
7. Optical resolution and expected image quality.
8. Lighting configuration and illumination uniformity.

The camera should therefore be selected backwards from the inspection requirement rather than by megapixel count alone.

## 6. Recommended Next Steps

The current results suggest that a new imaging dataset is not immediately necessary. The next stage should focus on connecting the defect-size analysis to the actual inspection setup.

The following experiment is also recommended within the current detection study:

**Evaluate a stride-4 (P2) feature level in RT-DETR for small defects, particularly Solder Ball.**

This has not yet been evaluated in the RT-DETR experiments and may provide useful evidence on whether additional small-object feature resolution improves localization.

For future image collection, capture metadata should also include the camera, lens, working distance, FOV, board identity, and lighting conditions so that imaging variables can be analysed separately from model performance.

## 7. Conclusion

The imaging analysis was conducted to determine whether FOV variation in the existing PCB dataset is a major factor affecting defect localization. The current results do not show a significant relationship between FOV-related sampling density and localization IoU. In contrast, defect pixel size shows a moderate positive relationship with localization quality.

Based on these findings, the immediate priority should not be to increase camera resolution or collect a new imaging dataset. Instead, the target inspection area, smallest relevant defect, required pixels per defect, and practical FOV should first be defined. These requirements can then be used to determine an appropriate camera, lens, and imaging configuration.

The findings should be considered as evidence from the current model and validation split rather than as a final conclusion about all possible imaging conditions.
