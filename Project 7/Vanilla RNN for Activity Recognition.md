## Vanilla Recurrent Neural Network(RNN)

**Vanilla RNNs** are the historical foundation of sequence models.  
  - They introduced the idea of **using past information to inform the present**.  
  - Even though they have limitations (vanishing gradients), they remain an essential starting point.  

**Reflection**

- **Shapes:** We have $(N, 128, 9)$ inputs and $(N,)$ labels for both train and test.  
- **Standardization:** Using train statistics only helps stabilize optimization and avoids leakage.  
- **Batching:** Our `DataLoader` yields `(batch, 128, 9)` tensors by default (**batch-first**).  
  <br>
Later, we will either **permute** to time-first `(128, batch, 9)` or create the RNN with `batch_first=True`. 

### Saving Preprocessed Datasets for Later Use

When we want to **reuse the exact same standardized dataset** without re-running all the preprocessing steps each time, we w**save the processed dataset** and load it later.

We save the standardized sequences and labels in two formats:

- **`.npy` (NumPy binary)**  
  - Keeps the exact array shape $(N, 128, 9)$, preserves `dtype`, and **loads very fast**.  
  - Ideal for deep learning workflows.  
  - Downside: not human-readable.

- **`.csv` (comma-separated values)**  
  - **Human-readable** and easy to inspect in Excel or even a text editor.  
  - Each sequence row is **flattened** from $128 \times 9$ into **1152 columns** + metadata columns.  
  - Slightly larger files, and requires reshaping back to $(128, 9)$ when loading.

**A note on `.npy` vs `.npz`**

When saving NumPy arrays, we often use two related formats:

- **`.npy`**: Stores a **single array** per file.  
  Example: `X_train_std.npy` contains only the training inputs $(N_{\text{train}}, 128, 9)$.  
  This is efficient, keeps the shape intact, and loads very quickly.  

- **`.npz`**: Stores **multiple arrays together** in one zipped container.  
  Example: `standardization_stats.npz` contains both `train_mean` and `train_std`.  
  Instead of creating two separate `.npy` files, we can keep them neatly grouped.

We chose `.npy` for the **datasets** (since they are large and independent),  
and `.npz` for the **normalization statistics** (small, logically grouped).  

**🔍 How do `.npy`, `.npz`, and `.csv` look inside?**

If you try to open these files in a text editor like Notepad, here’s what you’ll see:

- **`.csv`**  
  - Human-readable text.  
  - Each row is numbers separated by commas.  
  - Example (flattened sequence with label):
    ```
    0.12, -0.45, 0.33, ..., -0.04, 3
    ```
    Easy to inspect, but shape information (128×9) is *lost* — we must reshape.

- **`.npy`**  
  - Binary format: appears as *nonsense characters* in Notepad.  
  - That’s because it encodes array shape, dtype, and values in an efficient way.  
  - Think of it as a “locked box”: not human-readable, but NumPy can open it instantly with `np.load("file.npy")`.

- **`.npz`**  
  - A *zipped collection* of multiple `.npy` arrays.  
  - In Notepad it also looks like gibberish.  
  - But inside, it’s neatly storing several arrays with names (like `train_mean`, `train_std`).  
  - We can access them with `data["train_mean"]` after loading.

### Define the Vanilla RNN Classifier

**Architecture (what we will build)**

We will build the simplest useful sequence classifier:

```text
Input sequence (batch, 128, 9)
│
▼
nn.RNN (hidden_size = H, num_layers = 1, batch_first = True)
│
├─ out: all hidden states (batch, 128, H)
└─ h_n: final hidden state (1, batch, H)
│
▼
Take the last hidden state h_T → (batch, H)
│
▼
Linear (H → 6 classes) → logits (batch, 6)
│
▼
CrossEntropyLoss (applies softmax internally)
```
**Key choices:**
- **Batch-first** = `True` so our inputs are `(batch, seq_len, input_size) = (B, 128, 9)`.
- **One recurrent layer**, **unidirectional**, **tanh** nonlinearity (default).
- We will start with a modest **hidden size** (e.g., `H = 32`) to keep CPU runtime low.

**Why take the *last* hidden state?**

For **sequence classification** (our HAR task), one sequence → one label.  
The **last hidden state** $h_T$ acts as a **summary** of the entire 128-timestep window.  
We map $h_T$ through a linear layer to obtain 6 logits (one per activity class).

**Loss function and outputs**

We will use **`nn.CrossEntropyLoss`**, which:
- expects **logits** of shape `(batch, num_classes)`,  
- expects **target class indices** in `{0, …, 5}`,  
- **includes softmax internally**, so we **do not** apply softmax before the loss.

**What hyperparameters matter here?**

- `hidden_size (H)`: memory capacity (e.g., 32 or 64 are good starters on CPU).
- `num_layers`: depth in the time dimension (we keep it `1` for now).
- `nonlinearity`: `"tanh"` (default) or `"relu"`; we’ll use `"tanh"` as a classic vanilla RNN.
- `bidirectional`: `False` here (we’ll keep it simple and fast).

## Limitations of a Recurrent Neural Network(RNN)

**Theory: Why do gradients vanish?**

When training RNNs, we use **Backpropagation Through Time (BPTT)**:  
- Gradients flow backward through all timesteps (here, 128).  
- At each step, the gradient is multiplied by terms from the chain rule.  

If those terms are **smaller than 1**, the product shrinks **exponentially** with the number of timesteps.  
This leads to the **vanishing gradient problem**:

- Early timesteps contribute almost nothing to the final weight update.  
- The RNN struggles to capture **long-term dependencies**.  
- Training focuses mostly on the most recent steps.  

This is one of the main reasons more advanced architectures (LSTM/GRU) were developed.

### Plot of Gradient Norms after one backward pass of an RNN model

![Plot of Gradient Norms after one backward pass of an RNN model](<Gradient norms of RNN models.png>)

> **Note on `fc.weight` and `fc.bias`:**  
> 
> - In our model, the RNN outputs the final hidden state $h_{128} \in \mathbb{R}^{32}$.  
> - The **fully connected (fc) layer** maps this hidden vector to 6 activity classes:  
>   $$
>   \text{logits} = W \cdot h_{128} + b
>   $$
>   where `fc.weight = W` (shape: 6×32) and `fc.bias = b` (shape: 6).  
> - During backprop, the gradient flows first into this FC layer, then into the hidden state, and finally back through all timesteps of the RNN.  
> - That’s why you see `fc.weight` and `fc.bias` among the parameters — they usually have **healthy gradient magnitudes**, since they are directly connected to the loss, while the **recurrent weights** (`rnn.weight_hh_l0`) are where the vanishing gradient problem shows up.

**Reflection**

- Notice how some parameters (like **hidden-to-hidden weights**) may have **much smaller gradients** than others.  
- If we repeated this over many updates, we’d often see gradients shrink towards **zero**, especially for early timesteps.  
- This is direct evidence of the **vanishing gradient problem** in vanilla RNNs. 

>This plot shows the gradient magnitudes for different parameter groups after one backward pass. Notice the recurrent weights (weight_hh_l0) dominate — the model is heavily adjusting how memory >flows through time.<br>
>But remember: this does not directly reveal the vanishing gradient problem. To really see vanishing gradients, we would need to track how the gradients decay timestep by timestep inside the
unrolled RNN. That’s what we’ll attempt next. This plot is a first diagnostic — it tells us where the updates concentrate, but not how far the gradients actually travel back in time.

**Measure how much the final loss “pulls” on each input timestep**

We now estimate the **timestep-wise sensitivity**:

$$
\left\| \frac{\partial \mathcal{L}}{\partial x_t} \right\|_2
$$

### Vanishing gradient vs. timestep plot 
![Plot of vanishing gradients over time](<Vanishing gradient plot.png>)
**What we plotted.**  
For each timestep $t \in \{1,\dots,128\}$, we computed the average gradient magnitude:
$$
\Big\|\tfrac{\partial \mathcal{L}}{\partial x_t}\Big\|_2,
$$
where the loss $\mathcal{L}$ is computed only at the **final** timestep ($t=128$).  

This measures: *“If input $x_t$ changes slightly, how much does the final loss change?”*


**Why early timesteps have tiny gradients**

An RNN updates its hidden state step by step:
$$
h_t = \tanh(W_{ih} x_t + W_{hh} h_{t-1} + b), \quad
\hat{y} = \mathrm{softmax}(W_{hy} h_{128}).
$$

Now imagine the chain rule in action:

- For $t=127$:  
  $$
  \tfrac{\partial \mathcal{L}}{\partial x_{127}}
  \;\approx\; \alpha \cdot \tfrac{\partial \mathcal{L}}{\partial h_{128}}
  $$
 where $\alpha$ is some slope from $\tanh$ and weights ($<1$ in size).  
  → Gradient still strong, because it’s only 1 step away.

- For $t=100$:  
  $$
  \tfrac{\partial \mathcal{L}}{\partial x_{100}}
  \;\approx\; \alpha^{28} \cdot \tfrac{\partial \mathcal{L}}{\partial h_{128}}
  $$
  → Same shrink factor applied **28 times**. Much smaller.

- For $t=1$:  
  $$
  \tfrac{\partial \mathcal{L}}{\partial x_{1}}
  \;\approx\; \alpha^{127} \cdot \tfrac{\partial \mathcal{L}}{\partial h_{128}}
  $$
  → Tiny. The signal has almost disappeared.

So the further back we go, the more multiplications of “numbers less than 1” we accumulate, and the gradient almost vanishes.

**What the curve shows**

- Near the end of the sequence, gradients are **large** → the model can learn from those inputs.  
- Earlier timesteps → gradients **decay toward 0** → the model effectively **forgets the distant past**.  
- This is the **vanishing gradient problem** in action: the network can’t use information from long ago because the learning signal fades away before it gets there.
- So even if the **first input $x_1$ was important**, the network can’t “hear” its echo during training — the gradient is too weak to adjust the weights.  

This is why vanilla RNNs **struggle with long-term memory** 

**Hidden State Dynamics and Early Convergence**

If 

$$
\left\| h_t - h_{t-1} \right\|
$$

quickly drops near zero and stays low, we’re **converging too soon**.
