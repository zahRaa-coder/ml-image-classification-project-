Test set: 279 images, 6 classes

| model                             | accuracy   | top3_accuracy   | macro_precision   | macro_recall   | macro_f1   |   inference_ms_per_image |
|:----------------------------------|:-----------|:----------------|:------------------|:---------------|:-----------|-------------------------:|
| Random Forest (HSV+HOG+LBP)       | 44.8%      | 79.9%           | 45.3%             | 42.6%          | 42.8%      |                    191.8 |
| XGBoost (HSV+HOG+LBP)             | 46.2%      | 81.7%           | 50.5%             | 44.7%          | 45.7%      |                     14.1 |
| MobileNetV2 (fine-tuned CNN)      | 81.7%      | 97.1%           | 80.6%             | 80.2%          | 80.0%      |                     18.2 |
| Hybrid (CNN embeddings + XGBoost) | 78.1%      | 97.5%           | 76.5%             | 75.7%          | 75.7%      |                     39   |