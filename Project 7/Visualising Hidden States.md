## Visualising Hidden States

**Theory — What are hidden states?**

When we pass a sequence through an RNN:

- Each timestep $t$ produces a **hidden state** $h_t$.  
- With `hidden_size = 32`, each $h_t$ is a 32-dimensional vector:
  $$
  h_t = \big[h_t^{(1)}, h_t^{(2)}, \dots, h_t^{(32)}\big].
  $$
- Over 128 timesteps, the model produces 128 such vectors:
  $$
  (h_1, h_2, \dots, h_{128}), \quad h_t \in \mathbb{R}^{32}.
  $$

This gives us a **temporal trajectory** in a 32-dimensional space. Together, all 32 neurons form a **high-dimensional trajectory** that summarizes the input sequence.  
The final hidden state $h_{128}$ is then used by the classifier to decide the activity label.

**Why visualize hidden states?**
- Hidden states are the **memory** of the RNN.  
- At each timestep $t$, the RNN updates its hidden state using:

$$
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
$$

- The final hidden state $h_T$ is used for classification, but the **trajectory** $(h_1, h_2, \dots, h_T)$ contains rich information.  

By visualizing $h_t$ across time, we can see:
- How the RNN organizes sensor signals internally.  
- Whether different activities produce distinguishable hidden trajectories.  
- Where the model might **lose or confuse information**.

### Extract Hidden States

** What the RNN actually outputs**

For a batch of $B$ sequences, the forward pass returns:

- **`out_seq`**: all hidden states across time, shape $= (B, 128, H)$  
  (here $B=96$, $H=32$).
- **`h_n`**: the final hidden state(s), shape $= (1, B, H)$.
- **`logits`**: class scores after the FC layer, shape $= (B, 6)$.

So, the RNN is really doing two jobs:
1) Building a **memory trajectory** $h_1 \dots h_{128}$.  
2) Passing the **last hidden state** into a classifier.

**What we’ll visualize**

- Pick one sequence from the batch.  
- Extract its hidden states $h_1 \dots h_{128}$.  
- Plot a few hidden dimensions over time (like a “neuron firing curve”).  
- Also plot the **hidden norm** $\lVert h_t \rVert_2$ as a compact summary of how the hidden vector evolves.

### Dimensionality Reduction: PCA / t-SNE on Hidden States

Humans can’t visualize 32D directly, so we use **dimensionality reduction**:
  - **PCA (Principal Component Analysis):**
    - Linear method.
    - Projects data into directions of maximum variance.
    - Fast and interpretable.
  - **t-SNE (t-distributed Stochastic Neighbor Embedding):**
    - Nonlinear method.
    - Preserves local structure (points close in high-dim remain close in 2D).
    - Slower, but can reveal clusters more clearly.

**Implementation Plan**

1. Take one batch of test sequences.  
2. Collect all hidden states $h_t$ across all timesteps and sequences:
   - Shape = $(B \times T, H)$ = $(96 \times 128, 32)$.  
3. Apply PCA → 2D scatter plot.  
4. Optionally apply t-SNE → 2D scatter plot.  
5. Color points by activity label.

**Understanding PCA Intuitively**

We used **PCA (Principal Component Analysis)** to reduce our hidden states (32-dimensional vectors) down to **2D points**.  
But how do we go from 32 numbers to just 2?

**Step-by-step intuition**

1. **Hidden states in 32D:**  
   - Each hidden state $h_t \in \mathbb{R}^{32}$ is like a vector with 32 coordinates.  
   - We have thousands of such vectors (from 96 sequences × 128 timesteps each).  

2. **Find directions of maximum variance:**  
   - PCA searches for new axes (directions) that capture the **most variation** in the data.  
   - The **first principal component (PC1)** = direction where the data varies the most.  
   - The **second principal component (PC2)** = direction of the next-most variation, orthogonal to PC1.

3. **Project onto 2D:**  
   - Each 32D hidden vector is projected onto these two new axes (PC1 and PC2).  
   - Now every hidden state can be represented as just two numbers:  
     $$
     \text{Hidden state} \; h_t \;\to\; (PC1\_score, \; PC2\_score).
     $$

**Analogy**

Imagine taking a **shadow of a 3D object** onto a 2D wall:  
- You lose one dimension, but the shadow still shows the **main structure**.  
- PCA works the same way: it chooses the “best wall” (projection) that preserves as much information as possible.

**What to take away**

- Each dot = one hidden state $h_t$ from our RNN.  
- Dots with the same activity label (color) may cluster together if the hidden states encode that activity consistently.  
- Unlike t-SNE, PCA uses **linear projections**, so it may miss some nonlinear structure.  
- But PCA is fast, deterministic, and gives interpretable axes (variance explained by PC1 and PC2).

**Understanding t-SNE Intuitively**

We used **t-SNE** to reduce our hidden states (32-dimensional vectors) down to **2D points** for visualization.  
But what do these dots really mean?

**Step-by-step intuition**

1. **High-dimensional neighbors (32D):**  
   - Each hidden state $h_t \in \mathbb{R}^{32}$ is a point in 32D space.  
   - t-SNE looks at pairs of points and asks: *“How close are they?”*  
   - It converts these distances into probabilities (closer = higher probability of being neighbors).

2. **Create a 2D world:**  
   - All points are placed in a 2D plane.  
   - Again, we define neighbor probabilities, but this time using a different formula (Student-t distribution, which spreads points out more).

3. **Match the two worlds:**  
   - The algorithm moves the 2D points around so that:  
     - If two hidden states were close in 32D, they end up close in 2D.  
     - If they were far apart in 32D, they don’t collapse together in 2D.  
   - This is done by minimizing a mismatch (called KL-divergence) between the two sets of probabilities.

**Analogy**

Think of it like **flattening a crumpled paper**:

- In 3D (the crumpled ball), two ink dots may be neighbors even if their x/y coordinates look far apart.  
- When we flatten the paper into 2D, we try to keep those neighbor dots still close to each other.  
- We don’t care about the *exact* axes — only about **who stays close and who separates**.

**What to take away**

- Each dot = one hidden state $h_t$ from our RNN.  
- Clusters in 2D suggest the RNN has learned to group certain activities together.  
- The axes (t-SNE dim 1, t-SNE dim 2) **don’t carry meaning** — what matters is the **proximity and clustering**.  
- Color = the activity label of the sequence.  
- If dots of the same color cluster together → the RNN hidden states are encoding useful activity-specific patterns.  
- If clusters overlap → the RNN is struggling to fully separate those activities.