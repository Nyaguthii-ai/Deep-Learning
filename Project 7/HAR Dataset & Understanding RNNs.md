## Exploring the HAR Dataset & Understanding RNNs

### Sequence Learning with Recurrent Neural Networks (RNNs)
In this project, we use the **UCI Human Activity Recognition (HAR) Dataset**, where smartphone motion sensors capture human activities like walking, sitting, and standing. 

Our goal is to build and explore **Recurrent Neural Networks (RNNs)** to classify these activities.

### Human Activity Recognition (HAR)

Human Activity Recognition (HAR) is about **classifying physical activities** (like walking, standing, sitting) from sensor signals collected by wearable devices such as smartphones or smartwatches.  

In this project, we will use the **UCI HAR dataset**, where:  
- 📱 A Samsung Galaxy S II smartphone was worn at the waist.  
- 👥 30 volunteers performed **6 activities**:  
  - WALKING  
  - WALKING_UPSTAIRS  
  - WALKING_DOWNSTAIRS  
  - SITTING  
  - STANDING  
  - LAYING  
- 📡 The smartphone recorded **accelerometer** (linear acceleration) and **gyroscope** (angular velocity) signals at **50 Hz** (50 readings per second).  

**Data representation**

Each activity segment was preprocessed into **windows of 2.56 seconds** (≈ 128 timesteps).  

- Each window contains:  
  - **9 raw signals** (3 axes each for *total acceleration*, *body acceleration*, and *gyroscope*).  
  - This gives us an input shape of **128 × 9** (128 timesteps × 9 features).  
- Each window is assigned **one label** (the activity being performed).  

So one training example looks like this:

$$
X \in \mathbb{R}^{128 \times 9}, \quad y \in \{1,2,3,4,5,6\}
$$

**Why does the time structure matter?**

Let’s compare three types of models:

- **MLP** (Fully Connected Network)
  - Treats inputs as a flat vector.  
  - Ignores order of timesteps.  
  - Example: Flatten $(128 \times 9) \to 1152$ features.  

- **CNN** (Convolutional Neural Network)  
  - Captures local patterns (sliding filters).  
  - Great for spatial data (like images).  
  - With 1D filters, CNNs can capture short-term dependencies, but not long-term memory.  

- **RNN** (Recurrent Neural Network) 
  - Processes inputs **sequentially**:  
    - At timestep $t$, it uses the current input $x_t$ *and* the hidden state $h_{t-1}$ from the past.  
    - This allows the model to **remember history** and capture temporal dynamics.  

**RNN core idea (high level)**

At each timestep $t$:

$$
h_t = f(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
$$

- $x_t$ = input at time $t$ (9 features in our case).  
- $h_{t-1}$ = hidden state from previous timestep.  
- $h_t$ = updated hidden state (the model’s memory at $t$).  

At the end of the sequence ($t = 128$):

$$
\hat{y} = \text{softmax}(W_{hy} h_{128} + b_y)
$$

This gives us the predicted activity (one of the 6 classes).

### Data Exploration & Preprocessing

**What files do we have?**

The HAR dataset comes with **raw inertial signals** for each windowed segment (2.56s ≈ 128 timesteps). For each split (`train/` and `test/`), we have nine text files under `Inertial Signals/`, plus label files:

- **Signals (each row = one 128-timestep window)**  
  - `body_acc_x_[split].txt`, `body_acc_y_[split].txt`, `body_acc_z_[split].txt`  
  - `total_acc_x_[split].txt`, `total_acc_y_[split].txt`, `total_acc_z_[split].txt`  
  - `body_gyro_x_[split].txt`, `body_gyro_y_[split].txt`, `body_gyro_z_[split].txt`  

- **Labels**  
  - `y_[split].txt` → activity label for each window (integers in $\{1,\dots,6\}$).  
  - (Optional) `subject_[split].txt` → subject ID (1–30) for each window.

Each signal file has shape **(N, 128)**. Stacking the 9 channels gives us a **multivariate sequence** of shape **$(N, 128, 9)$**.

> One sample (one window) therefore looks like: $X \in \mathbb{R}^{128 \times 9}$ with a single label $y \in \{1,2,3,4,5,6\}$.

**What does “preprocessing” mean here?**

We’ll do a few minimal, transparent steps:

1. **Load** the nine signals per split and **stack** them along the last axis to get $(N, 128, 9)$.
2. **Load labels** as integers (we’ll also show **one-hot encoding** for pedagogy, even though PyTorch’s `CrossEntropyLoss` expects class indices, not one-hot).
3. **Normalize** features (optional but recommended): compute **per-channel** mean/std on the **training set only**, and standardize both train and test:
   $$
   X' = \frac{X - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}
   $$
   This keeps magnitudes comparable across channels and helps stable training on CPU.
4. **Prepare tensors** for PyTorch and preview a couple of examples.

**Quick note on one-hot vs. class indices**

- **Class indices** (0–5) are what `nn.CrossEntropyLoss` expects.  
- **One-hot vectors** (length 6) are useful for metrics/plots or if we ever use `nn.BCEWithLogitsLoss` in a multi-label setup.  
We’ll keep **class indices** as our ground truth and optionally show how to get one-hot when needed.

**Visual sanity checks**

We’ll:
- Print shapes (`(N, 128, 9)` for inputs, `(N,)` for labels).
- Show the **first window** as a small matrix preview and its label name.
- Plot short **sequence snippets** for two activities (e.g., **WALKING** vs **SITTING**) to build intuition about temporal patterns.

**Preparing shapes for RNNs**

Our DataLoaders will typically yield **batch-first** `(batch, seq_len, input_size)` = `(B, 128, 9)`.  
PyTorch’s `nn.RNN` by default expects **time-first** tensors:
$$
(\text{seq\_len}, \text{batch}, \text{input\_size}) = (128, B, 9).
$$
We’ll use `x.permute(1, 0, 2)` before the forward pass (or set `batch_first=True` in the RNN if we prefer batch-first).

At the end of the sequence ($t=128$), we use the last hidden state to make a prediction:

$$
\hat{y} = \text{softmax}(W_{hy} h_{128} + b_y)
$$

**Intuition**

- The RNN is like a **rolling summary** of everything seen so far.  
- At each timestep, the hidden state **compresses the past + current input** into one value.  
- By the end of 128 steps, the last hidden state $h_{128}$ represents the whole sequence. 

**Time-first vs batch-first**

RNNs in PyTorch expect inputs in one of two formats:

- **Time-first (default):**  
  $(\text{seq\_len}, \text{batch}, \text{input\_size})$  
  Example: $(128, 32, 9)$  
  → At timestep 1, the RNN processes *all 32 sequences* in parallel.  

- **Batch-first (optional):**  
  $(\text{batch}, \text{seq\_len}, \text{input\_size})$  
  Example: $(32, 128, 9)$  
  → More intuitive (batch dimension comes first).  

We can choose either:
- Use **time-first** (default) and call `.permute(1,0,2)` to swap axes.  
- Or create the RNN with `batch_first=True`.  

**Intuition**

- In **time-first**, we conceptually “march through time” one step at a time, updating all sequences in the batch together.  
- In **batch-first**, we conceptually “line up full sequences” for each sample, then let the RNN handle the time loop internally.  
- Both are equivalent — it’s just about how we arrange the input tensor.

**Sequence classification (our case: HAR)**

- Input: a window of **128 timesteps × 9 features**.  
- Output: **1 activity label** for the whole window.  
- Method:  
  - We only take the **last hidden state** $h_{128}$.  
  - Pass it through a classifier (linear layer + softmax).  
  - One probability distribution over the 6 activities.  

Formally:

$$
\hat{y} = \text{softmax}(W_{hy} h_{128} + b_y)
$$


**Sequence labeling (other tasks)**

- Example: Part-of-Speech tagging in NLP, speech recognition, gesture recognition.  
- Input: a sequence of tokens/sounds/frames.  
- Output: **one label for each timestep**.  
- Method:  
  - Use every $h_t$.  
  - Classifier produces $\hat{y}_t$ for each $t$.  

Formally:

$$
\hat{y}_t = \text{softmax}(W_{hy} h_t + b_y), \quad t=1,\ldots,128
$$

**Visualization**

Think of it this way:

- **Sequence classification (HAR):**  
  Sequence of hidden states $h_1, h_2, \ldots, h_{128}$ → only last $h_{128}$ used → one label.  

- **Sequence labeling (e.g., POS tagging):**  
  Sequence of hidden states $h_1, h_2, \ldots, h_{128}$ → each $h_t$ → label $\hat{y}_t$.  
