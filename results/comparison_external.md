External test set: 720 TrashNet images (120/class max) - different source, never seen in training

| model                             | accuracy   | top3_accuracy   | macro_precision   | macro_recall   | macro_f1   |
|:----------------------------------|:-----------|:----------------|:------------------|:---------------|:-----------|
| Random Forest (HSV+HOG+LBP)       | 21.5%      | 62.2%           | 24.5%             | 21.5%          | 15.9%      |
| XGBoost (HSV+HOG+LBP)             | 25.4%      | 64.0%           | 31.0%             | 25.4%          | 22.1%      |
| MobileNetV2 (fine-tuned CNN)      | 58.5%      | 89.6%           | 57.7%             | 58.5%          | 57.6%      |
| Hybrid (CNN embeddings + XGBoost) | 58.6%      | 90.7%           | 57.9%             | 58.6%          | 57.9%      |