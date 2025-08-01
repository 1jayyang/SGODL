# SGODL: Statistically Guided Optimal Distribution Learning for Semi-Supervised Medical Image Segmentation

## 📄 Abstract

Semi-supervised learning has significantly advanced medical image segmentation under limited annotation scenarios. However, mainstream semi-supervised medical image segmentation (SSMIS) methods based on consistency regularization and pseudo-labeling are vulnerable to feature distribution shifts and noisy pseudo labels caused by internal covariate shift (ICS). Such shifts disrupt semantic clustering in the feature space and lead to pseudo-label instability across training iterations, ultimately resulting in suboptimal model performance.

In this work, we present the first systematic analysis of the impact of ICS in semi-supervised semantic segmentation, and propose a **Statistically Guided Optimal Distribution Learning (SGODL)** framework that explicitly models feature distributions to mitigate ICS. Specifically, we introduce an **Optimal Distribution Learning (ODL)** mechanism that adaptively models feature distribution parameters from unlabeled data to suppress variance fluctuations in feature space. Additionally, we propose a **Latent Distribution Difference Compensation (L2DC)** strategy that captures and leverages distributional differences between labeled and unlabeled data in the latent space, guiding the network beyond the segmentation patterns constrained by the limited labeled data and substantially enhancing its generalization to unseen cases.

Extensive experiments on three challenging medical image segmentation benchmarks demonstrate the superiority of our method over state-of-the-art approaches. We expect this work to offer valuable insights into the SSMIS community from a new perspective.

## 📊 Datasets

We evaluate our method on the following widely-used 3D medical image segmentation datasets:

- **BraTS2019** – Brain Tumor Segmentation  
- **LiTS2017** – Liver Tumor Segmentation  
- **Pancreas-NIH** – Pancreas Segmentation

## 📂 Code Availability

👉 **Full code will be released upon paper acceptance/publication.**

Stay tuned for updates, and feel free to watch or star the repository!

## 📬 Contact

For questions or collaborations, please contact us via GitHub Issues or email (to be provided upon paper publication).
