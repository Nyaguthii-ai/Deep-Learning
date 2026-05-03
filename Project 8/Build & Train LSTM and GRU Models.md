## Build & Train LSTM and GRU Models

We will train **three sequence models** on exactly the same inputs so our comparison is fair and easy to interpret:

- **Vanilla RNN (baseline):** a minimal recurrent model that processes each 7-day window and predicts the next temperature. This gives us a *reference point* for accuracy, stability, and speed.
- **Long Short-Term Memory (LSTM) networks: ** an RNN with **gates** and a **cell state** designed to keep useful information alive longer. We expect it to handle patterns that a plain RNN might forget.
- **Gated Recurrent Unit (GRU) networks:** a lighter-weight gated RNN that often matches LSTM performance with fewer parameters and slightly faster training.

Our task is **single-step forecasting**: given a window of 7 past days, predict **tomorrow’s** temperature.

**Why LSTM/GRU vs. Vanilla RNN?**

Vanilla RNNs struggle with **vanishing gradients**—the learning signal fades as we backprop through many timesteps.  
**LSTM** and **GRU** add **gates** that control information flow:

- **Remember useful context:** Gates preserve signal over longer spans (better long-term memory).
- **Forget stale context:** Forget/update gates reset memory when patterns shift.
- **Stabilize gradients:** Gated paths reduce repeated multiplications by small Jacobians, mitigating vanishing.
- Both layouts are valid; **the math inside RNN/LSTM is identical**. If a layer expects time-first, we convert a batch-first tensor with:  
  $$X_{\text{time-first}}=\operatorname{permute}\!\left(X_{\text{batch-first}}, (1,0,2)\right)$$
  i.e., $(B,\,S,\,F)\rightarrow(S,\,B,\,F)$.

> **Why batch-first here?** After NB01, our dataset is naturally “one row = one window,” so reading $(B,\,S,\,F)$ is simpler and avoids constant permutation. We still know how to switch if a layer needs time-first.

**LSTM — adding a regulated memory (cell state + gates)**

An **LSTM** augments the hidden state $h_t$ with a **cell state** $c_t$ (longer-term memory) and three **gates** that control what to keep, write, and reveal. At each step:
$$
\begin{aligned}
i_t &= \sigma\!\big(W_{xi}x_t + W_{hi}h_{t-1} + b_i\big) &&\text{(input gate; how much new info to write)}\\
f_t &= \sigma\!\big(W_{xf}x_t + W_{hf}h_{t-1} + b_f\big) &&\text{(forget gate; how much old info to keep)}\\
o_t &= \sigma\!\big(W_{xo}x_t + W_{ho}h_{t-1} + b_o\big) &&\text{(output gate; how much memory to expose)}\\
\tilde{c}_t &= \tanh\!\big(W_{xc}x_t + W_{hc}h_{t-1} + b_c\big) &&\text{(candidate content)}\\[4pt]
c_t &= f_t \odot c_{t-1} \;+\; i_t \odot \tilde{c}_t \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$
Here $\sigma(\cdot)$ is the logistic sigmoid (outputs in $(0,1)$), and $\odot$ is element-wise multiplication.

**Why this helps with long memory:**  

If $f_t \approx 1$ and $i_t \approx 0$ for several steps, the update becomes $c_t \approx c_{t-1}$ — the **cell state travels forward nearly unchanged**. Its gradient also travels with **minimal shrinking**, a property sometimes called the “constant error carousel.” We can then **preserve** important information across many steps and decide **when** to update or expose it via $i_t$ and $o_t$.

> Intuition: LSTM memory $\approx$ a **whiteboard with knobs** — we can *erase* (forget), *write* (input), and *reveal* (output) information on demand.

A plain RNN tends to **overweight the last 1–2 days** (recency bias) and can forget the start of the window. An LSTM can **learn to keep** a smoothed representation of the week (via $c_t$) and update it **selectively** with $i_t, f_t$. In practice, that often yields **steadier** next-day forecasts, especially around transitions (e.g., cooling after a hot spell).

** Alternative: GRU (lighter gating)**

We now swap the recurrent cell to a **GRU**. A GRU is a **gated** RNN like LSTM but with **fewer parameters**: it merges gates and does not maintain a separate cell state. In practice, GRUs often train a bit **faster** and can achieve **similar accuracy** to LSTMs on many tasks.

**Parallel setup for fair comparison**

- Inputs remain batch-first: $ (\text{batch}, \text{seq\_len}, \text{input\_size}) = (B, 7, 1) $.
- Core: $\texttt{nn.GRU}$ with $\texttt{batch\_first=True}$, $\texttt{input\_size}=1$, $\texttt{hidden\_size}=64$, $\texttt{num\_layers}=1$.
- Head: take the last time step’s hidden state $h_{\text{last}}$ and apply a $\texttt{Linear}(64 \rightarrow 1)$ to predict the next day.

**GRU intuition in a line :**
GRU uses **update** and **reset** gates:
- **Update gate** $z_t$ blends old memory and new candidate: it decides **how much to keep** from $h_{t-1}$.
- **Reset gate** $r_t$ controls **how much past** to consider when forming the new candidate $\tilde{h}_t$.

> Intuition: $z_t$ is a **gate for updating** the state; when $z_t$ is small, we keep most of $h_{t-1}$.  
> $r_t$ is a **gate for resetting** past influence when computing the candidate.

Compared to LSTM’s three gates and cell state $c_t$, GRU **simplifies** the mechanism, often with **similar modeling power** for short to medium contexts.

**Reflection**

**When might GRU match LSTM despite fewer parameters?**

- When the dataset’s **effective temporal dependencies** are **short to medium** (e.g., our 7-day window), GRU’s simpler gating can capture the needed context without a separate cell state.  
- When we need **faster training** or have **smaller models** (CPU-only, tight time budget), GRU’s lower parameter count can help us converge quickly.  
- When the signal is **noisy** and we don’t benefit much from very long memory, GRU often performs on par with LSTM.

>**RNN, LSTM, GRU — quick comparison**

| Model | Memory mechanism | Gates | Param cost (rough) | Strengths | Limits | Problems addressed |
|------|-------------------|-------|--------------------|-----------|--------|--------------------|
| **RNN** | Hidden state $h_t$ only | – | $W_{ih}, W_{hh}$ ⇒ baseline | Simple, fast | Vanishing gradients; short effective memory | – |
| **LSTM** | Hidden $h_t$ + cell $c_t$ | input, forget, output | ~**4×** RNN (four affine blocks) | Long-range memory; gradient flow via $c_t$ | Heavier (params/compute) | Vanishing gradients, long dependencies |
| **GRU** | Hidden $h_t$ only | update, reset | ~**3×** RNN | Good compromise: fewer params than LSTM, strong memory | Still heavier than RNN | Vanishing gradients, medium/long deps |

*Rule of thumb:* RNN < GRU < LSTM in capacity/robustness and in parameter count/compute. We’ll keep hidden sizes modest so runs remain CPU-friendly.
