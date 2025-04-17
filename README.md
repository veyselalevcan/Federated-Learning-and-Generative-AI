# Federated-Learning-and-Generative-AI
# 🧠 GAN-Based SCADA Attack Simulation & Anomaly Detection

This repository presents a research project focused on simulating **False Data Injection Attacks (FDIA)** on SCADA systems using **Generative Adversarial Networks (GANs)** and detecting them with deep learning-based anomaly detection models.

> 🔬 Dataset: [Secure Water Treatment (SWaT) Testbed – July 20, 2019](#dataset-overview)

---

## 📌 Project Overview

- Simulate **multi-dimensional cyberattacks** using WGAN-GP on real SCADA sensor data (24 features).
- Inject synthetic attacks into normal SCADA data to generate realistic anomaly-rich datasets.
- Use anomaly detection models like **Autoencoders** and **LSTM-AE** to detect attacks in both centralized and federated setups.
- Visualize and analyze attack separation using feature distributions, t-SNE projections, and reconstruction error histograms.

---

## 📊 Visualization Summary

<div align="center">
  <img src="feature value dist normal vs gan.png" alt="GAN vs Normal Visualizations" width="100%" />
</div>

**Figure Explanation:**

| Panel | Description |
|-------|-------------|
| **Left: Feature Value Distribution** | Comparison of mean feature values per sample between Normal (blue) and GAN-generated attack (red) samples. Clear separation indicates statistical anomaly. |
| **Center: t-SNE Projection** | Dimensionality reduction highlights clear separation between Normal (blue circles) and Attack (red crosses) samples, suggesting effective attack simulation. |
| **Right: Reconstruction Error (MSE)** | Histogram of Autoencoder reconstruction errors. A threshold (e.g., 0.28) is used to classify potential anomalies. Most GAN attacks exceed this threshold. |

---

## 📁 Dataset Overview

**Source:** SWaT Testbed – Water Treatment ICS/SCADA Testbed  
**Collection Date:** July 20, 2019  
**Normal Operation:** 12:35–14:50 (GMT+8)  
**Attack Window:** 15:08–16:16  
**Notable Attacks:**

1. **FIT401 spoofing:** Value changed from 0.8 → 0.5 (affects UV401 actuation).
2. **LIT301 spoofing:** Level changed from 835 → 1024 (forces T301 underflow).
3. **P601 forced ON:** Fills raw water tank.
4. **MV201 + P101 simultaneous ON:** May overflow T301.
5. **MV501 OFF:** Blocks RO tank drainage.
6. **P301 OFF:** Interrupts ultrafiltration (UF) process.

---

## 🧬 Methodology

1. **GAN-Based Simulation (WGAN-GP):**
   - Input: Normal samples from SWaT dataset.
   - Output: Synthetic attack samples that mimic FDIA behavior.

2. **Attack Injection:**
   - Generated attacks are injected into normal sequences to simulate contaminated sensor streams.

3. **Anomaly Detection Models:**
   - **Autoencoder (AE)**
   - **LSTM-AE**
   - **Transformer (optional extension)**

4. **Federated Learning Setup (optional):**
   - Clients receive partitioned datasets with local attacks.
   - Models are trained collaboratively without raw data sharing.

---

## 🧠 Model Evaluation

- **Reconstruction Loss (MSE):** Used to compute anomaly score per sample.
- **Detection Thresholding:** Empirically chosen from the validation set (e.g., 0.28).
- **Performance Metrics:** Precision, Recall, F1, AUC.

---

## 📦 Repository Structure


---

## 📖 References

- SWaT Dataset (iTrust, Singapore University of Technology and Design):  
  [https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)

- WGAN-GP: Gulrajani et al., *Improved Training of Wasserstein GANs*  
  [https://arxiv.org/abs/1704.00028](https://arxiv.org/abs/1704.00028)

---

## ✨ Future Work

- Integrate more attack types (DDoS, MITM).
- Extend to **cross-site federated learning** using additional ICS datasets (e.g., BATADAL).
- Use **VAE-GAN** or **Diffusion Models** for higher realism in attack simulation.

---

## 🛡️ Disclaimer

This project is for academic research purposes only. The SWaT dataset and generated attacks should not be used for malicious purposes.

---

## 🤝 Contributing

PRs and collaborations are welcome, especially for federated learning support and real-time monitoring tools.

---


