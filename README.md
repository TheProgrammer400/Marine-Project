# Shrimp Biomass Estimation using Contrastive Learning

This project focuses on estimating the biomass (mass) of shrimps using computer vision and deep learning.  
Instead of treating all marine species equally, the system is designed specifically for shrimps, which are commercially and economically important.

The approach uses **contrastive learning** to first learn robust visual representations of shrimps and then applies these representations to downstream tasks such as shrimp detection and mass estimation.

---

## Project Motivation

- Not all marine species are harvested or used as resources.
- Shrimp farming and fisheries require accurate estimation of shrimp biomass for:
  - Yield prediction
  - Resource management
  - Sustainability
- Manual weighing is time-consuming and impractical at scale.

This project aims to automate shrimp mass estimation from images.

---

## Core Idea

1. Learn **what a shrimp looks like** using self-supervised contrastive learning.
2. Reuse the learned backbone for:
   - Detecting multiple shrimps in a single image
   - Estimating the mass of each shrimp
3. Sum individual masses to estimate total biomass.

---

## Architecture Overview

### 1. Contrastive Pretraining (SimCLR-style)

- Backbone: ResNet50 (ImageNet initialized)
- Projection Head: MLP (used only during contrastive training)
- Training Data: Unlabeled shrimp images
- Objective: Learn shrimp-specific visual representations

Image → Backbone → Projection Head → Contrastive Loss


After pretraining:
- Projection head is discarded
- Backbone weights are saved and reused

---

### 2. Shrimp Detection (Multi-object Images)

- Uses the contrastively pretrained backbone
- Detection head predicts bounding boxes for individual shrimps

Image → Backbone → Detection Head → Shrimp Bounding Boxes


---

### 3. Shrimp Mass Estimation (Regression)

- Each detected shrimp is cropped
- Cropped shrimp is passed through the backbone
- A regression head predicts shrimp mass


Shrimp Crop → Backbone → Regression Head → Mass (grams)


Total biomass is calculated by summing individual shrimp masses.

---

## Dataset

### Contrastive Learning Dataset
- Only shrimp images are required
- No labels are used
- Images may contain background objects (hands, trays, etc.)
- Disease-based folders are merged during contrastive training

Example sources:
- Shrimp disease image datasets
- Aquaculture shrimp image collections

### Important Notes
- Class imbalance does not affect contrastive learning
- Occasional noisy or low-quality images are acceptable
- Directories and non-image files are filtered out

---

## Training Details

- Loss Function: NT-Xent (contrastive loss)
- Temperature: 0.2
- Batch Size: 64 (or higher if resources allow)
- Training stops when loss plateaus (not based on absolute value)

A final loss around ~1.0 is considered healthy for this setup.

---

## Implementation Highlights

- PyTorch-based implementation
- Custom Dataset class for contrastive learning
- Robust file filtering to avoid directory loading errors
- Backbone reused across tasks (no repeated contrastive training)

---

## Future Work

- Improve shrimp detection accuracy using instance segmentation
- Incorporate size-to-weight calibration using real measurements
- Extend the system to other aquaculture species
- Deploy as a real-time monitoring tool

---

## Conclusion

This project demonstrates how self-supervised contrastive learning can be effectively used to build a scalable and robust shrimp biomass estimation system.  
The learned representations are reusable across multiple tasks, making the approach efficient and extensible.

