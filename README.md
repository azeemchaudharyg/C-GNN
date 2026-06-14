# Context-Aware Graph Neural Network for Skin Lesion Classification

<p align="left">
  <a href="https://dl.acm.org/doi/proceedings/10.1145/3748522" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/ACM_SAC-2026-blue" alt="Conference"></a>
  <a href="https://doi.org/10.1145/3748522.3779958" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/DOI-10.1145/3748522.3779958-red" alt="DOI"></a>
  <a href="https://github.com/azeemchaudharyg/C-GNN/blob/main/notebooks/GNN_Multimodal.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Framework-PyTorch_Geometric-purple" alt="Implementation"></a>
  <a href="#" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
</p>

Official repository accompanying the paper: **"Context-Aware Graph Neural Network for Skin Lesion Classification"** 

**Authors:** Muhammad Azeem, Saqib Nazir, Amr Ahmed, and Ardhendu Behera  

**Institution:** Edge Hill University, Ormskirk, Lancashire, UK 

---

## Abstract
The accurate classification of skin lesions, particularly melanoma, is vital for the early detection and effective treatment of skin cancer. Although deep learning models such as convolutional neural networks (CNNs) have achieved remarkable success in dermoscopic image analysis, they often overlook valuable structured metadata (e.g., patient demographics, lesion location, and type) that provide essential diagnostic context. We present a graph-driven multimodal framework that jointly models visual and metadata information for skin lesion classification. Our approach uses a frozen CNN backbone to extract deep visual representations via dual pooling (mean and max), which are concatenated with encoded metadata and partitioned into subspaces. These subspaces are treated as nodes within a graph, where a graph neural network (GNN) captures intra-sample dependencies between feature subspaces and clinical attributes to refine lesion representations. Experiments on four public benchmarks: ISIC2024, HAM10000, PAD-UFES-20, and HIBA, demonstrate consistent performance gains over several state-of-the-art (SOTA) approaches, with relative accuracy improvements ranging from +0.5% to +8.8% across datasets. The results highlight the potential of graph-based modeling of metadata and image features to build more robust and clinically informed skin cancer classifiers.

---

## Key Contributions
* **Graph-Driven Multimodal Fusion:** Introduces a novel framework that integrates CNN-derived visual subspaces and clinical patient metadata into a unified graph, propagating context-aware features for more discriminative lesion representation.

* **Inductive GNN Formulation:** Employs a principled feature-to-graph transformation via systematic subspace partitioning, backed by an inductive message-passing mechanism that allows the model to generalize seamlessly across unseen datasets without fixed graph constraints.

* **Comprehensive Validation:** Demonstrates state-of-the-art robustness across four public benchmarks, achieving average relative gains of +1.31% in accuracy, +1.06% in precision, +2.32% in recall, and +1.24% in F1-score.

---

## Method Architecture

The structural workflow of the Context-Aware GNN operates in three primary phases:
1. **Regional Feature Embedding:** Extracting patch-level descriptors using a deep CNN backbone.
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
