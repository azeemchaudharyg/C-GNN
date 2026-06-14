# Context-Aware Graph Neural Network for Skin Lesion Classification

<p align="center">
  <a href="https://dl.acm.org/doi/proceedings/10.1145/3748522" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/ACM_SAC-2026-blue" alt="Conference"></a>
  <a href="https://doi.org/10.1145/3748522.3779958" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/DOI-10.1145/3748522.3779958-red" alt="DOI"></a>
  <a href="https://github.com/azeemchaudharyg/C-GNN/blob/main/notebooks/GNN_Multimodal.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Framework-PyTorch_Geometric-purple" alt="Implementation"></a>
  <a href="LICENSE" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
</p>

Official repository accompanying the paper: **"Context-Aware Graph Neural Network for Skin Lesion Classification"** 

**Authors:** Muhammad Azeem, Saqib Nazir, Amr Ahmed, and Ardhendu Behera  

**Institution:** Edge Hill University, Ormskirk, Lancashire, UK 

---

## Abstract
Traditional deep learning approaches treat skin lesion patches as isolated visual entities, frequently ignoring surrounding tissue structures and spatial cross-correlations that are vital for accurate dermatological diagnosis. In this paper, we present a novel **Context-Aware Graph Neural Network (GNN)** framework designed explicitly to model high-level spatial context alongside local lesion topologies. By structuring dermoscopic image regions as heterogeneous graphs, our model establishes edge dependencies that catch nuanced boundaries and structural anomalies common to malignant formations. This relational spatial learning approach improves the network's discriminative power, yielding noticeable improvements over standalone convolutional pipelines for multi-class skin lesion categorization.

---

## Key Contributions
* **Context-Aware Graph Modeling:** Introduces a custom graph-structured representation strategy that links regional skin features with macro-level context matrices.
* **Topological Feature Aggregation:** Leverages message-passing graph networks to capture inter-region edge affinities, allowing for superior border and texture evaluation.
* **Robust Multi-Class Mapping:** Delivers highly accurate classifications for multi-class dermatological profiles under diverse acquisition protocols.

---

## Method Architecture

The structural workflow of the Context-Aware GNN operates in three primary phases:
1. **Regional Feature Embedding:** Extracting patch-level descriptors using a deep convolutional or transformer backbone.
2. **Graph Construction:** Mapping spatial regions as nodes and connecting structural boundaries using geometric distance thresholding.
3. **Contextual Aggregation:** Executing message-passing operations across the constructed graph to generate context-aware predictions.

<p align="center">
  <img src="images/gnn_architecture.png" width="650" alt="Context-Aware GNN Architecture Diagram"><br>
  <em>Figure: Overview of the Context-Aware Graph Neural Network pipeline for lesion classification.</em>
</p>

---

## Installation & Setup

This framework leverages **PyTorch** and **PyTorch Geometric (PyG)** to run optimized graph message-passing operations.

### 1. Environment Initialization
```bash
conda create -n context-gnn python=3.10 -y
conda activate context-gnn
