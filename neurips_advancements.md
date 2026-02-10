# Recent Advancements in fMRI Task Prediction (NeurIPS 2024-2025)

This document summarizes key advancements from NeurIPS 2024 and 2025 that relate to predicting task-based fMRI activation from resting-state data, particularly focusing on zero-shot generalization and the Natural Scenes Dataset (NSD).

## 1. ZEBRA: Zero-Shot Cross-Subject Generalization
**Paper:** *ZEBRA: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding* (NeurIPS 2025)

*   **Core Idea:** Disentangles subject-specific vs. semantic-specific components of fMRI signals using adversarial training.
*   **Advancement:** Unlike linear Procrustes alignment (Hyperalignment), ZEBRA extracts "subject-invariant" features. 
*   **Performance:** Achieves state-of-the-art zero-shot decoding on NSD without requiring any subject-specific task data or linear alignment steps for the target subject.

## 2. SwiFUN: Swin fMRI UNet Transformer
**Paper:** *SwiFUN: Predicting Task Activation Maps from Resting-State fMRI* (NeurIPS 2024)

*   **Core Idea:** Utilizes the full 4D spatiotemporal dynamics of resting-state fMRI instead of static connectivity matrices.
*   **Advancement:** Leverages a **Swin 4D fMRI Transformer (SwiFT)** to map temporal sequences of rest directly to 3D task activation maps.
*   **Result:** Captures complex, non-linear patterns that static correlation matrices (fingerprints) miss, showing up to a 27% improvement in task prediction accuracy over previous CNN-based models.

## 3. NeuroMamba: State-Space Foundation Models
**Paper:** *NeuroMamba: A State-Space Foundation Model for Functional MRI* (NeurIPS 2025)

*   **Core Idea:** Applies a Mamba (State-Space Model) backbone for direct sequence modeling of 4D whole-brain fMRI.
*   **Advancement:** Shifts away from small-scale training to **large-scale self-supervised pre-training** (Foundation Models).
*   **Result:** Treats fMRI as a large-scale spatiotemporal sequence problem, allowing for better universal brain representations that generalize across arbitrary subjects.

## 4. Comparison with Current Approach

| Feature | Current Plan (CHA + Ridge) | NeurIPS 2024/2025 Advancements |
| :--- | :--- | :--- |
| **Resting-State Input** | Static Connectivity (Correlations) | **4D Spatiotemporal Sequences** (Transformers/Mamba) |
| **Alignment Method** | Linear Procrustes / Basis Projection | **Adversarial Disentanglement** / Foundation Pre-training |
| **Model Complexity** | Linear Ridge Regression | **Non-linear Transformers** / Universal Encoders |
| **Generalization** | Specific to NSD subjects | **Zero-shot across arbitrary subjects** via pre-training |

## Recommendations for Implementation

1.  **Upgrade the Encoder:** Move from Ridge regression to a non-linear **MLP or Transformer encoder** (as suggested in Approach C of the PLAN.md).
2.  **Contrastive Alignment:** Implement a contrastive loss (similar to ZEBRA) to align shared-space brain representations more tightly with semantic features (e.g., CLIP embeddings).
3.  **Dynamic Features:** If static connectivity (CHA) performance plateaus, consider extracting features from resting-state temporal sequences using a pre-trained backbone like **SwiFT** instead of just parcel-wise correlations.
