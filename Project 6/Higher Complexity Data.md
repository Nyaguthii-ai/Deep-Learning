### Higher Complexity Data

Key challenges when working with complex data i.e plants:
  - **Intra-class variation** (same plant looks different at different growth stages).
  - **Background noise** (soil, weeds, overlapping plants).
  - **Class imbalance** (some species are underrepresented).

The Dataset prepares us to handle **messier datasets** than clean academic ones.

Offer a perfect starting point to explore **transfer learning** with pretrained models like ResNet or AlexNet. 

These models were originally trained on ImageNet, a massive dataset of 1.2M images across 1,000 classes (dogs, cats, cars, plants, etc.).

ImageNet models expect 224×224 input — this is the standard resolution used during their training.
- By resizing images to 224×224, we match the expected input size, allowing us to reuse pretrained weights effectively.
- 224×224 is also a good balance: detailed enough to capture plant structures, but not too large to slow down CPU-based training.

**Why Use ImageNet Normalization (Mean/Std)?**

- Pretrained CNNs not only expect 224×224 input, but also **specific pixel statistics**:  
  - **Mean:** `[0.485, 0.456, 0.406]`  
  - **Standard Deviation:** `[0.229, 0.224, 0.225]`
- These values are computed from the **entire ImageNet dataset** and represent the typical RGB distribution across millions of natural images.

**Why normalize?**
- Ensures input images are **centered** (mean 0) and **scaled** (std ~1).
- Matches the distribution that pretrained weights were optimized on — crucial for transfer learning to work well.
- Without this, the pretrained features would misinterpret brightness/contrast and hurt performance.

Ref: <br>
https://discuss.pytorch.org/t/discussion-why-normalise-according-to-imagenet-mean-and-std-dev-for-transfer-learning/115670<br>
https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2<br>
https://discuss.pytorch.org/t/normalizing-with-imagenet-mean-and-std-vs-normalizing-with-my-own-datasets-mean-and-std/94953<br>


**Visualization Goals**

- Display a **grid of sample images** (one per class) to familiarize ourselves with the data.  
- Plot a **class distribution chart** to identify imbalance (e.g., Loose Silky-bent vs. Common wheat).  
- This helps anticipate challenges (e.g., some species may dominate the dataset and bias the model).

**Why Transfer Learning Now?**

Instead of **training CNNs from scratch** on every dataset:
- We can use **pretrained networks** (e.g., ResNet, AlexNet) trained on **ImageNet (1.2M images, 1000 classes)**.
- These models already **learned general visual features** (edges, shapes, textures) that transfer well to new tasks.
- We only need to **fine-tune or replace the final layers** for our target classes 
- Smaller dataset (~4k images) benefits from pretrained features.
- Faster convergence and better accuracy with limited data.
