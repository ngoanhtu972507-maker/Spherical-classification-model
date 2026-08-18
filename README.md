# Sphere Manifold Classification on CIFAR-10

## Overview
This repository implements an image classification system for the CIFAR-10 dataset, transitioning from a standard Euclidean feature space to a Spherical Manifold representation. The architecture integrates a customized ResNet18 backbone with a Spherical Classifier, utilizing Robust Spherical Prototype Extraction (RSPE) and Principal Geodesic Analysis (PGA).

## Core Training Pipelines

The system is structured into two distinct operational pipelines to manage Euclidean and non-Euclidean computations.

### 1. Forward Pipeline (Feature Representation & Inference)
The forward pass maps raw images to manifold logits.
- Image Processing: 32x32 input images are passed through a customized ResNet18 backbone (conv1 reduced to 3x3, stride set to 1, MaxPool removed to preserve spatial resolution).
- Feature Extraction: The network outputs a 512-dimensional feature vector in Euclidean space.
- Radial Projection: The 512-D vector undergoes L2 normalization and is projected onto a sphere manifold of radius R.
- Distance Computation: The geodesic distance is calculated between the projected feature vectors and K class prototypes located on the sphere. To ensure numerical stability, the argument of `arccos` is clamped between -1.0 and 1.0.
- Logit Generation: Class scores are computed via weighted prototype aggregation. The scores pass through a LogSumExp function and are scaled by a temperature parameter to produce the final predictive logits.

### 2. Backpropagation Pipeline (Manifold Optimization & Updating)
The backward pass handles dual-space optimization (Riemannian and Euclidean).
- Composite Loss Calculation: The objective function evaluates a standard Cross-Entropy Loss combined with a Separation Loss. The separation loss pushes prototypes apart and is controlled by a progressively increasing coefficient.
- Tangent Space Mapping: Gradients for the manifold-bound parameters (prototypes) cannot be updated via standard addition. They are mapped to the tangent space at the Fréchet mean using the Log Map.
- Riemannian Optimization: Parameters are updated within the tangent space using a Riemannian optimizer (e.g., RiemannianAdam), then mapped back to the spherical manifold using the Exponential Map.
- Backbone Optimization & Unfreezing: Euclidean gradients update the ResNet18 weights using a progressive strategy. Early stages freeze the backbone to stabilize the spherical head. Subsequent stages unfreeze layers sequentially from deep (`layer4`) to shallow (`conv1`), utilizing differential learning rates to preserve pretrained representations.
- Cool-down Phase: In the final epochs, strong data augmentations (CutMix, MixUp, RandomErasing) are disabled, and a cosine annealing schedule scales the learning rate down for convergence stability.

## Analytical Components

### Robust Spherical Prototype Extraction (RSPE)
RSPE initializes and refines class prototypes while mitigating outlier influence. It computes distance reliability scores, applies adaptive soft trimming to down-weight anomalies, and iterates via Riemannian optimization until convergence thresholds are met.

### Principal Geodesic Analysis (PGA)
PGA serves as the manifold-aware equivalent of PCA for dimensionality reduction and visualization. It calculates the Fréchet mean, projects data to the tangent space, applies SVD, and evaluates the explained variance ratio.

## Performance Metrics & Evaluation
The model is benchmarked on the following expected ranges:
- Test Accuracy: 75-82%
- Parameter Count: 11M - 12M
- FLOPs (Single Sample): 500M - 600M
- Inference Latency (Batch 64): 5-12 ms

## Environment & Hardware Configuration

The implementation supports cross-platform hardware acceleration. It leverages MPS natively for efficient local execution on Apple Silicon architectures (such as the M2 Pro), while maintaining full compatibility with Linux environments (like Ubuntu) via CUDA or standard CPU fallback mechanisms. 

### Installation
```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision geoopt scikit-learn matplotlib plotly thop
```

### Execution
The primary entry point is `Sphere_dist_reliability_prototype.ipynb`.
Execution flows sequentially from environment initialization through PCA/PGA exploration, ending with the 100-epoch training pipeline and interactive 3D manifold visualization.
