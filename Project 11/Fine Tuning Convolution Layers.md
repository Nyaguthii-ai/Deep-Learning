## Fine-Tuning Convolutional Layers

Fine-tuning is the process of *partially unfreezing* a pretrained model so that some of its layers can adapt to the new dataset.  
Instead of keeping the entire convolutional backbone frozen, we’ll allow a few of its deeper layers to learn again, but at a **very small learning rate**.  
This helps the model slightly reshape its internal representations while still retaining most of its pretrained knowledge.

**Why fine-tune?**

Feature extraction works well when our target dataset is **similar** to the source dataset (e.g., both contain natural images).  
However, if our dataset has new textures, shapes, or styles that differ from ImageNet, we might want to *fine-tune* the deeper layers so the model adapts to these new characteristics.

Fine-tuning lets us:
- Refine learned representations for our specific task.
- Achieve better accuracy for certain hard-to-classify categories.
- Explore how pretrained features evolve when allowed to learn slightly.

Previously, we **froze all pretrained layers** of ResNet18 and trained only the **new classifier head**.  
That approach was called **feature extraction** because the pretrained backbone acted as a fixed feature generator — it transformed each image into a rich, high-level representation, and only the classifier learned to map those representations to our new classes.

Now, we take a step forward and begin **fine-tuning**.

**Feature Extraction vs. Fine-Tuning**

We can summarize the transition as follows:

> **Before (Feature Extraction)**  
> Frozen backbone → Trained head  
>
> **Now (Fine-Tuning)**  
> Partially unfrozen backbone → Trained head  

This adaptation lets the model learn subtle differences — new textures, object shapes, or lighting conditions — that were not present in the original ImageNet training data.

**Why Only Unfreeze the Deeper Layers?**

The early layers of a CNN learn very **generic features**, like:
- edges,  
- corners,  
- simple color gradients.  

These are universal across images and do not need retraining.

The **deeper layers**, however, learn **task-specific features**, such as the shape of animal faces or car parts.  
These are the layers that we want to adapt — just a little — to suit our new task.

**The Risk: Catastrophic Forgetting**

Fine-tuning must be done carefully.  
If the learning rate is too high, the pretrained weights can be overwritten — causing the model to “forget” the general features it had learned from ImageNet.  
This phenomenon is called **catastrophic forgetting**.

That’s why, during fine-tuning, we use:
- a **small learning rate** (e.g., $1\times10^{-4}$ or lower),
- careful layer selection (usually just the top block),
- and limited epochs.

**ResNet-18 Architecture Overview**

| Stage | Name | Typical Output Size | Description | Trainable Status |
|:------:|:------|:------------------:|:-------------|:----------------|
| 1 | `conv1` + `bn1` + `relu` + `maxpool` | 64×56×56 | Initial convolutional stem extracting low-level edges and textures | ❌ Frozen |
| 2 | `layer1` | 64×56×56 | First residual block — still low-level spatial features | ❌ Frozen |
| 3 | `layer2` | 128×28×28 | Mid-level features — shapes, textures, and contours | ❌ Frozen |
| 4 | `layer3` | 256×14×14 | Higher-level patterns and object parts | ❌ Frozen |
| 5 | `layer4` | 512×7×7 | Deepest convolutional block — object-level abstractions | ✅ To be Unfrozen |
| 6 | `avgpool` + `fc` | 512→`num_classes` | Global pooling + final classifier | ✅ Unfrozen |

**Why unfreeze only `layer4`?**
- Earlier layers capture **generic visual features** (edges, color blobs) that are transferable across datasets.  
- Deeper layers (especially `layer4`) encode **dataset-specific semantics**, so fine-tuning this part helps the model adapt to our Caltech-10 classes without overfitting or retraining the entire network.  
- This approach is called **“partial fine-tuning”** — it balances generalization and adaptation.

### Optimizer, Learning Rate, and Scheduler

A practical recipe is to use **differential learning rates**:
- A **lower LR** for the unfrozen pretrained layers (here, `layer4`, e.g., $1\times 10^{-4}$).  
- A **slightly higher LR** for the new classifier head (e.g., $1\times 10^{-3}$), since it was not pretrained and can move faster.

We keep the same **criterion** as before: `nn.CrossEntropyLoss()` for multi-class classification.

Optionally, we add a simple scheduler such as `StepLR` to **decay the learning rate** after a couple of epochs. This helps stabilize training and can yield a small boost in generalization.

### Evaluating and Comparing Results

We now evaluate our **fine-tuned model** on the held-out test set and compare it directly with the **feature-extraction (FE)** baseline from NB02.

What we will examine:

- **Overall accuracy** on the test set.  
- **Per-class precision, recall, and F1**, to see which categories improved.  
- **Confusion matrices (side-by-side)** for FE vs Fine-Tuning (FT), to visualize where confusions decreased or shifted.

What should we expect?

- Classes that differ more from typical ImageNet categories may **benefit more** from fine-tuning (since deeper filters adapt).  
- Some classes may see **marginal** changes if FE already separated them well.  
- Improvement is **not guaranteed** — it depends on **domain similarity**, dataset size, and regularization.  
- If FT underperforms FE, that can signal **overfitting** or too-aggressive unfreezing/learning rates.

**How to read these comparisons**

 - If **Accuracy** and **Macro F1** increase from FE → FT, fine-tuning helped overall.  
 - Inspect **ΔF1 by class** to see *where* the gains came from. Often, classes with textures or shapes less common in ImageNet benefit more.  
 - The **side-by-side confusion matrices** show whether certain confusions shrank after fine-tuning (rows look “sharper” along the diagonal).

**Why improvement varies**

 - If our dataset is **close to ImageNet** (natural object photos), FE might already be near-optimal, so FT yields **small gains**.  
 - If some categories are **under-represented** or **visually ambiguous**, FT may not help much without augmentation or more data.  
 - If FT underperforms FE, we may need:
   - **Smaller LR** on unfrozen layers,  
   - **Fewer unfrozen layers** (e.g., only part of `layer4`),  
   - **Regularization** or **augmentation**,  
   - **Fewer epochs** to avoid overfitting.

### Visualizing Feature Adaptation

To understand **what fine-tuning changed**, we will look inside the network and compare **pre–fine-tuning** (feature extraction, FE) with **post–fine-tuning** (FT) along the last residual stage, `layer4`.

We will visualize two things:

1. **Activation maps** from `layer4` for the *same* image before and after fine-tuning.  
   - If fine-tuning helped, we often see **sharper, more localized activations** over the object of interest or its parts.  
   - If nothing changed, activations may look very similar to the FE case.

2. **Grad-CAM** heatmaps for the **predicted class** before and after fine-tuning.  
   - Grad-CAM highlights the **spatial regions** that contribute most to the model’s decision.  
   - If FT improved performance, Grad-CAM may shift attention from background to **object regions** or attend more strongly to **discriminative parts**.

**Reading the activation maps (mean over channels)**

 - If the **FT map** looks more concentrated on the object or its parts than the FE map, that suggests `layer4` has adapted to **more discriminative regions**.  
 - If both maps look similar, the feature extraction representation may already be strong for this sample.

**Reading the Grad-CAM overlays**

 - Grad-CAM highlights **where** the model looks to make its decision.  
 - After fine-tuning, attention often shifts **away from background** and toward **object boundaries or distinctive textures**.  
 - If the predicted class changes from FE → FT (and becomes correct), check whether FT’s CAM aligns better with **true object regions**.

**Why this matters**

 - These visualizations connect **performance gains** to **representational changes**.  
 - Fine-tuning does not rewrite the whole network; it **nudges higher-level features** to better fit our dataset.  
 - When the target dataset diverges from ImageNet, these nudges can be the difference between **ambiguous** and **decisive** internal representations.

### Reflection: When to Fine-Tune (and When Not To)

Fine-tuning is a powerful tool — but not always the best choice.  
Here, we step back to think about **when it makes sense** to unfreeze pretrained layers and **when it’s better** to keep them frozen.


🧠 When **Feature Extraction** Is Sufficient
Feature extraction (keeping the backbone frozen) often works best when:
- The **dataset is small**, and we risk overfitting if we train too many parameters.  
- The **target domain is similar** to ImageNet — for example, natural photographs of everyday objects.  
- We need **fast training** and **stable convergence**.  
- The goal is to build a **strong baseline** with minimal computation.

> Think of it as using an already trained “visual encoder” and only teaching the model how to map features to new labels.

⚙️ When **Fine-Tuning** Helps
Fine-tuning pays off when:
- We have **enough data** to adjust deeper weights meaningfully.  
- The **domain differs** from ImageNet (e.g., microscopic images, artworks, X-rays, or unusual textures).  
- We need to **squeeze out extra accuracy** beyond the frozen baseline.  
- We are studying **representation specialization** — how features evolve for new tasks.

> In such cases, fine-tuning allows the deeper convolutional filters to “shift” toward the new dataset’s structure — learning domain-specific color palettes, shapes, or boundary cues.


**⚖️ Trade-offs**
| Aspect | Feature Extraction | Fine-Tuning |
|:--|:--|:--|
| **Training time** | Short | Longer |
| **Data requirement** | Low | Higher |
| **Risk of overfitting** | Low | Higher |
| **Adaptability** | Limited | Greater |
| **Computation cost** | Low | Higher |

Fine-tuning can yield **better adaptation** but at the cost of **extra computation** and potential **loss of generality** — the network may “forget” some of its broad ImageNet knowledge if learning rates are too high.
