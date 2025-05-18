# AI Project – Empirical Privacy Evaluation of (ϵ, δ)-Differentially Private Models for Fraud Detection

This repository contains the code for the final project of the [MSc in Artificial Intelligence](https://www.mtu.ie/courses/crkarti9/) at the Munster Technological University (MTU) in Cork, Ireland.

## Project Description

Fraud detection systems play a critical role in the financial sector, where protecting sensitive data is as important as identifying illicit activities. While machine learning and deep learning models have proven effective in detecting anomalies, their use often requires access to confidential data, raising privacy concerns. This thesis explores the impact of (ε, δ)-Differential Privacy on both utility and empirical privacy in fraud detection models, with a focus on the underexplored δ parameter. Two main model architectures were investigated: a Long Short-Term Memory (LSTM) and an XGBoost classifier. Theywere evaluated under different DP mechanisms, including DP-SGD, PATE, and a differentially private oversampling technique using MST-based synthetic data generation. Membership Inference Attacks (MIA) was used to measure privacy leakage across a range of (ε, δ) configurations and dataset sizes. The results show that the privacy parameter ε has a dominant effect on both utility and privacy. For both models, privacy stabilises for δ < 10−4, suggesting that lower δ values offer little additional protection in practice. Interestingly, differential privacy affected the deep learning model’s utility more significantly, especially under strong privacy constraints, often leading to underfitting and reduced utility. In contrast, XGBoost maintained high performance even under tight privacy guarantees. This work provides an empirical evaluation of how (ε, δ)-differential privacy affect the empirical privacy of models in fraud detection and shows that incrementing the training set size increases the privacy even when the performance does not.

## Authors

- [Albert Salles](https://github.com/albertsalles4)
- [Dr Diarmuid Grimes](https://scholar.google.com/citations?user=oki58f0AAAAJ&hl=en) - Project Supervisor