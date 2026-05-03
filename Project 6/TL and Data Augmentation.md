### Transfer Learning and Data Augmentation 

**Why Data Augmentation Now?**

Despite the accuracy boost, we still face challenges:

- **Overfitting signs:** Validation accuracy plateaued while training accuracy kept rising.  
- **Class imbalance:** Some species (e.g., *Black-grass*) remain underrepresented → prone to misclassification.  
- **Environmental noise:** Background, lighting variations, and overlapping features introduce variability.

**Data augmentation** helps by artificially increasing dataset diversity:
- Random flips, rotations, and color adjustments simulate real-world variability.
- Forces model to learn **robust features** rather than memorizing pixel patterns.

Data augmentation artificially expands the training dataset by applying random transformations to existing images. This simulates new variations without requiring extra manual labeling and helps models generalize better to unseen data.

By applying augmentation during training, we **expose the model to these variations** on-the-fly.

**Common Augmentation Techniques**

1. **Spatial Transforms**
   - Random horizontal/vertical flips
   - Random rotations
   - Random resized crops (zoom-in effects)

2. **Color Jitter**
   - Random brightness, contrast, saturation, hue adjustments

3. **Noise & Blur (optional)**
   - Gaussian noise or blur to simulate camera/environment variability

**Impact on Training**

- **Benefits:**
  - Reduces overfitting by increasing data diversity
  - Encourages learning of **invariant features** (e.g., shape over color)

- **Trade-offs:**
  - Slightly longer training (augments images each epoch)
  - Augmentation strength must be balanced (too strong can harm accuracy)

**Conceptual Flow**

```text
Raw Image → Random Augmentation (flip/rotate/jitter) → Tensor Conversion & Normalize → Model Input
```

We now **apply data augmentation** during training to improve generalization. The key idea:

- **Training set**: Use **randomized augmentations** (different every epoch).
- **Validation/Test set**: Use **deterministic preprocessing** (no augmentation) to ensure fair evaluation.

**Why Two Pipelines?**

- Training pipeline **simulates variety** (helps robustness).
- Test/Validation pipeline **preserves ground-truth distribution** (avoids evaluation bias).

We expected augmentation to boost accuracy, but result may be similar to the non-augmentend model. This happens for a few reasons:

- Augmentation makes training harder: the model must learn to handle **extra variability** (flips, rotations, color shifts).  
- We only trained **10 epochs** — not enough to adapt to this harder task.  
- The backbone was **frozen**; pretrained features may not align perfectly with augmented images.  
- Some augmentations may not match real-world variations, introducing **noise rather than useful diversity**.

**Key takeaway:**  
Augmentation helps when realistic and paired with proper training (more epochs or slight fine-tuning). In practice, tuning augmentation strength is part of model development — not every augmentation automatically improves performance.

**Next Steps: Improving Augmentation Performance**

**How can we improve?**

1. **Train Longer**
   - Augmentation introduces more variability, so the model needs **more epochs** (e.g., 15–20) to converge.
   - Use **early stopping** to avoid overfitting while allowing enough time for learning.

2. **Moderate Augmentations**
   - Reduce **rotation range** (e.g., ±10° instead of ±20°).
   - Lower **color jitter** strength (slight brightness/contrast changes instead of strong shifts).
   - Ensure augmentations mimic **real-world variations** in seedlings (not unrealistic distortions).

3. **Fine-Tune Backbone**
   - Instead of freezing the entire ResNet18 backbone, unfreeze the **last residual block(s)**.
   - This allows the model to adapt pretrained features to the **specific patterns** of seedlings.

4. **Progressive Augmentation**
   - Start training with **original images** (stabilize learning), then gradually introduce augmentations.
   - Helps balance **stability vs generalization** during training.
