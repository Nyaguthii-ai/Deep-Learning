## Evaluating GANs & Common Challenges

Unlike classifiers, GANs lack a straightforward metric such as accuracy or F1-score.
Their losses do not directly indicate visual quality. Instead, evaluating a generative model means judging how realistic, diverse, and stable its outputs are — through both quantitative and qualitative lenses.

Their goal is to **generate data indistinguishable from reality** — but *how do we quantify “realistic”?*

**No Explicit Likelihood or Reconstruction Loss**

Unlike **Variational Autoencoders (VAEs)**, which include a reconstruction loss that measures how well the output matches the input,  
GANs **do not model data likelihood** directly.  
There is no single scalar metric that tells us whether the generator has truly captured the data distribution.

**The Subjectivity of “Realism”**

What does *realistic* even mean in the context of art?

- A perfectly blended color palette?  
- A composition that mimics brush strokes of real paintings?  
- Or something that evokes human emotion or aesthetic appeal?

Unlike datasets containing faces or everyday objects, **artistic realism is subjective and multidimensional**.  
Perception of realism depends on human taste, artistic conventions, and cultural interpretation — making objective evaluation nearly impossible.

**Qualitative Aspects of GAN Evaluation**

Because of these inherent challenges, we rely heavily on **qualitative indicators** to understand how well a GAN performs.  
Three aspects are most commonly used:

- **Visual Quality** – Sharpness, coherence, and structural integrity of generated images.  
  Does each image *look like* it belongs to the target artistic style?

- **Diversity** – Variety across generated samples.  
  Does the generator produce diverse brush strokes, color schemes, and compositions?

- **Mode Coverage** – Representation of multiple “modes” or categories in the data.  
  For example, if we trained on six art styles, does the generator produce outputs from all of them — or collapse into one dominant mode?

**The GAN Evaluation Triangle**

These three aspects — **realism**, **diversity**, and **stability** — often exist in **tension**.  
Improving one can sometimes degrade the others.  

```text
           +---------------------+
           |      REALISM        |
           |   (Image Quality)   |
           +---------------------+
                   /\
                  /  \
                 /    \
       DIVERSITY ------ STABILITY
   (Variety)              (Training Balance)
```

### Understanding Mode Collapse

**What is mode collapse (and why we care)?**  
In a balanced art dataset, we expect our generator $G$ to produce **diverse** paintings: different styles, palettes, brush textures, and compositions.  
**Mode collapse** happens when $G$ maps many different latent vectors $z \sim \mathcal{N}(0, I)$ to **very similar outputs** (e.g., the same color blobs or a single recurring composition).  
- If the Discriminator $D$ provides **strong gradients** for one particular kind of image, $G$ can learn a “shortcut”: produce only that kind of image to fool $D$.  
- This **imbalance** in the adversarial game reduces diversity: $G$ “wins” locally but **fails** to represent the full data distribution.

**How to interpret this**

- **Visual inspection:**  
  If many samples look **very similar** (same colors, same blobs, same layout) across different $z$ values, that’s a red flag for **mode collapse**.

- **Heuristics:**  
  - **Pixel std across the batch** (scalar):  
    Higher values usually indicate **greater variety** across samples.  
  - **Mean pairwise cosine similarity** (scalar):  
    Lower values usually indicate **less redundancy** (i.e., images are more distinct).

**Mitigate:**  
  - Adjust learning rates (often lower $D$’s LR or raise $G$’s LR slightly).  
  - Use **label smoothing**, **instance noise**, or **data augmentation** for $D$.  
  - Try **mini-batch discrimination** (explicitly encourages sample diversity).  
  - Increase **model capacity** or move to stronger architectures (e.g., StyleGAN variants) when resources allow.  
  - Ensure **balanced data** so $G$ is exposed to varied modes.

### Comparing Generated vs. Real Images

We can directly compare **generated** images with **real artwork** from the WikiArt dataset.  

Our goal here is not to assign a numeric score but to **visually and intuitively** assess how close the generator comes to the true artistic data distribution.

Visually inspect aspects like:
- Brush stroke textures  
- Color palettes  
- Composition and edges  
- Artistic coherence

- *“Can we visually separate real vs. fake paintings?”*  
  Observe whether generated samples blend seamlessly or still appear artificial.  
  If most viewers can’t easily tell, the GAN is achieving **visual plausibility**.

- *“What artistic details still differ?”*  
  Look for subtle aspects: brush strokes, edges, fine gradients, or compositional balance.  
  GANs often capture **color distribution and rough form**, but may lack **texture coherence** or **semantic depth**.

**Summary Insight**

| Aspect | Real WikiArt | Generated (DCGAN) | Observations |
|:--------|:--------------|:------------------|:--------------|
| Color Palette | Natural, balanced | Sometimes over-saturated | Needs better regularization |
| Texture Detail | Fine brushwork | Smoothed or repetitive | Limited texture realism |
| Composition | Structured | Slightly abstract or repetitive | Potential mode collapse |
| Overall Impression | Artistic and varied | Convincing but less nuanced | Promising realism; room for diversity |

This final comparison highlights how **GAN evaluation** ultimately blends **quantitative metrics** and **human aesthetic judgment** — reminding us why GANs sit at the intersection of **science and creativity**.