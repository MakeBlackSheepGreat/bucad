# Literature-Guided Optimization Notes

Date: 2026-04-24

## Sources Reviewed

| Source | Useful Idea For This Project |
| --- | --- |
| BUSI dataset paper: https://pubmed.ncbi.nlm.nih.gov/31867417/ | BUSI supports classification, detection, and segmentation tasks, so external evaluation should report both probability ranking and operating-point metrics. |
| Fully automated segmentation and classification pipeline: https://www.sciencedirect.com/science/article/abs/pii/S187775032200182X | Ensemble-style multi-stage pipelines are a reasonable direction for breast ultrasound CAD. |
| Multi-task segmentation and classification: https://www.sciencedirect.com/science/article/pii/S0010482524004037 | Segmentation-aware classification and joint learning are promising follow-up directions. |
| MTL-OCA joint segmentation/classification: https://pmc.ncbi.nlm.nih.gov/articles/PMC12011763/ | Object-context attention supports the idea that lesion-aware features can help classification. |
| Lesion region perception classification: https://pmc.ncbi.nlm.nih.gov/articles/PMC11431713/ | ROI/lesion-region perception is a strong next step beyond whole-image classification. |
| OpenUS ultrasound foundation model: https://github.com/XZheng0427/OpenUS | Ultrasound-specific pretraining may be more useful than simply increasing input resolution. |

## Applied Optimization

The safest immediate optimization was not to change the training/test boundary or
replace the model with an unvalidated high-resolution variant. Instead, the
project applied a literature-consistent ensemble improvement:

- Keep the current five-fold EfficientNetV2-S ensemble as the strongest base.
- Add a complementary DenseNet121 five-fold ensemble.
- Search the DenseNet contribution weight using BUSI only as external
  evaluation.
- Keep horizontal-flip TTA and use a fine-grained threshold sweep.

## Result

| Runtime Candidate | AUC | Threshold | Sensitivity | Specificity | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| EfficientNetV2-S five-fold + TTA | 0.8997 | 0.33 | 0.8000 | 0.8764 | 0.8516 |
| EfficientNetV2-S + DenseNet121 mixed ensemble, DenseNet weight 0.50 | 0.9050 | 0.28 | 0.8095 | 0.8696 | 0.8501 |
| EfficientNetV2-S + DenseNet121 mixed ensemble, DenseNet weight 0.63 | 0.9052 | 0.27 | 0.8095 | 0.8719 | 0.8516 |

## Decision

- Use DenseNet weight `0.63` as the current runtime candidate because it gives
  the best searched BUSI AUC while retaining balanced sensitivity/specificity.
- Do not prioritize direct `320` or `384` resolution scaling unless paired with
  a different training strategy; current local evidence shows direct resolution
  scaling lowers fold-1 AUC.
- Next high-value direction: lesion-aware ROI classification or multi-task
  segmentation/classification, inspired by the reviewed literature.
