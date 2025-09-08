# DNLP SS25 Final Projet
-   **Group name:** The Tokenizers
    
-   **Group code:** DNLP G16

-   **Group repository:** https://github.com/bilalahmed0060/deep-nlp-BERTified
    
-   **Tutor responsible:** 	Frederik Hennecke
    
-   **Group team leader:** Muhammad Usman Khanzada
    
-   **Group members:** 
  
# Setup Instructions
### Initial Setup

1. Clone the repository:
```bash
git clone https://github.com/bilalahmed0060/deep-nlp-BERTified.git
cd deep-nlp-BERTified/
```
2. Run the GWDG setup script:
```bash
./setup_gwdg.sh
conda activate dnlp
```
 
## 5) Paraphrase Generation:
### 1) Install extra packages
```bash
pip install optuna
```
### 2) Project structure & data
Your repo should look like this:
```bash
pip install optuna
```project_root/
├─ bart_generation.py                  
├─ data/
│  ├─ etpc-paraphrase-train.csv
│  └─ etpc-paraphrase-generation-test-student.csv
├─ predictions/
│  └─ bart/                            # will be created if you generate test outputs
└─ ...
```
### 3) Execution modes
#### A) Supervised-only (MLE)
Trains BART with padding-masked cross-entropy and saves the best dev checkpoint to best_supervised_bart_model.pt.
```bash
python bart_generation.py \
  --do_supervised \
  --supervised_epochs 8 \
  --seed 11711
```
After this, later runs will automatically start from `best_supervised_bart_model.pt` if present.

#### B) RL (SCST) on penalized BLEU (starting from best supervised if available)
Fine-tunes with self-critical sequence training using the penalized BLEU reward.
```bash
python bart_generation.py \
  --use_gpu \
  --use_rl --rl_epochs 5 \
  --rl_lr 2e-6 --rl_lambda 0.3 --rl_batch_size 8 --rl_max_len 50 \
  --seed 11711
```
#### C) Quality-guided RL (optional)
Adds a controllable reward blend (semantic, syntactic, lexical). Example weights:
```bash
python bart_generation.py \
  --use_gpu \
  --use_rl --rl_epochs 6 \
  --quality_weights "0.7 0.2 0.1" \
  --rl_lr 2e-6 --rl_lambda 0.3 --rl_batch_size 8 --rl_max_len 50 \
  --seed 11711
```
#### D) Hyperparameter Optimization (HPO) + RL
Runs Optuna to tune RL hyperparameters on the dev penalized BLEU, then applies the best params for a full RL run.
```bash
python bart_generation.py \
  --use_gpu \
  --do_supervised --supervised_epochs 8 \
  --use_rl --rl_epochs 5 \
  --hpo --hpo_trials 20 --hpo_rl_epochs 2 \
  --seed 11711
```
### 4) CLI quick reference
```bash
--do_supervised            Run supervised MLE training before RL
--supervised_epochs        Epochs for supervised phase (default: 10)

--use_rl                   Enable RL (SCST) fine-tuning
--rl_epochs                RL epochs (default: 2)
--rl_lr                    RL AdamW learning rate (default: 2e-6)
--rl_lambda                Weight for RL vs MLE blend (default: 0.3)
--rl_max_len               Max generation length during RL (default: 50)
--rl_batch_size            Batch size for RL (default: 8)

--quality_weights          e.g., "0.7 0.2 0.1" (semantic, syntactic, lexical)

--hpo                      Run Optuna HPO before final RL
--hpo_trials               Number of trials (default: 20)
--hpo_rl_epochs            RL epochs per trial (short budget; default: 1)
--hpo_timeout              Optional global timeout for HPO (seconds)

--use_gpu                  Use CUDA if available
--seed                     Global random seed (default: 11711)

```
# Methodology

This task focuses on improving paraphrase generation performance using a customized BART architecture. Our primary goal is to explore, implement, and evaluate various training strategies and reward functions within a unified and extensible pipeline that supports both baseline and advanced methods via configurable command-line arguments. The overarching methodology is to systematically extend the standard BART model into a flexible research platform for paraphrase generation tasks, incorporating reinforcement learning-based fine-tuning, quality-guided rewards, and hyperparameter optimization. Through this approach, we aim to enhance the diversity and semantic fidelity of generated paraphrases while maintaining alignment with evaluation metrics such as penalized BLEU.
## Problem & Objective

We tackle **paraphrase generation** with the goal of producing fluent outputs that **preserve meaning** while **discouraging copying** from the source. Training is aligned with evaluation using a **penalized BLEU** objective, used both as the RL reward and for dev model selection:

$$
R(\text{ref}, \text{inp}, \text{hyp}) = \text{BLEU}(\text{ref}, \text{hyp}) \times \frac{100 - \text{BLEU}(\text{inp}, \text{hyp})}{52}
$$

We compute BLEU with **SacreBLEU** (`effective_order=True`) to mitigate length and low-count artifacts.


## Dataset
- We were provided with etpc-paraphrase-train.csv for training data and etpc-paraphrase-generation-test-student.csv as test data.
- We split the training data into 80/20 to obtain the dev data.


## Model & Data Encoding

**Backbone:** `facebook/bart-large` (seq2seq)
**Inputs:** Each example is serialized as
```ini
encoder_input = sentence1  <SEP>  sentence1_segment_location  <SEP>  paraphrase_type_ids
```
where `<SEP>` is the tokenizer’s sep_token (or </s> fallback).

- **Targets:** `sentence2` (gold paraphrase).

- **Padding masking:** target pads are ignored via labels `[labels == pad_id] = -100`, ensuring <pad> tokens do not contribute to the cross-entropy, otherwise there will be noise in the loss.

## Baseline: Supervised Fine-Tuning (MLE)
We first fine-tune BART with standard cross-entropy on (input → target) pairs. This yields a fluent, stable initialization:
- Optimizer: **AdamW**
- Checkpointing: save `best_supervised_bart_model.pt` whenever dev penalized BLEU improves.
- Rationale: MLE provides strong fluency and stabilizes the subsequent RL phase.

## Improvement: Self-Critical Sequence Training (SCST) on Penalized BLEU
We adopt **SCST** to directly optimize our task reward. For each batch:

- **Greedy baseline (`greedy_decode`)**: decode deterministically (no sampling).  
- **Sampled rollout (`sample_decode`)**: decode with exploration  
  `do_sample=True, top_p=0.9, top_k=50, temperature=1.5, min_length=8, repetition_penalty=1.1`.  
- **Per-sample reward**: sentence-level SacreBLEU with smoothing composed into penalized BLEU as above.  
- **Advantage**:
   
  **_A = R_sample - R_greedy_**

- **Policy gradient**: compute $\log p_\theta(y_{\text{sample}} \mid x)$ via teacher forcing (`sequence_logprob`) and minimize **_-A.logp_**.     

#### Stability controls
- **Loss mixing (MLE stabilizer):**  
  _L = λ_rl · L_PG + (1 − λ_rl) · L_MLE,**λ_rl = 0.3**_
- **Advantage normalization** with variance floor (`max(std, 0.1)`); skip PG if `std < 1e-6`.  
- **Gradient clipping:** `clip_grad_norm_(…, 1.0)`.  
- **Optimizer state persistence:** a single AdamW optimizer is created once in the driver and passed into each RL epoch so momentum/second-moment statistics carry across epochs.  

#### Why it helps
- The reward aligns directly with how we evaluate (faithfulness × novelty), explicitly penalizing copies.  
- SCST’s **self-baseline** reduces variance by subtracting the greedy trajectory’s reward.  

---

### Optional: Quality-Guided Reward
Inspired by [Bandel et al., 2022], we add an **optional quality-guided reward**:  

- **Semantic similarity** (SacreBLEU vs. reference, proxy).  
- **Syntactic variation** (length-difference proxy).  
- **Lexical variation** (overlap penalty proxy).  

Users set weights via `--quality_weights "sem syn lex"` or a JSON config.  
This keeps training lightweight (no extra models) while enabling interpretable trade-offs between adequacy and diversity.  

---

### Hyperparameter Optimization (HPO) for RL (Optional)
We include an Optuna (TPE) search that maximizes dev penalized BLEU over RL-critical knobs:

- **Search space**  
  - `rl_lr` ∈ [1e-7, 5e-5] (log)  
  - `lambda_rl` ∈ [0.1, 0.7]  
  - `rl_max_len` ∈ {40, 50, 60, 70, 80}  
  - `rl_batch_size` ∈ {4, 8, 16}  

- **Protocol**  
  - Each trial runs short RL (`--hpo_rl_epochs`, default 1) on a deterministic subset of train for speed.  
  - Model is reset to the pre-HPO snapshot before each trial.  
  - We evaluate on the same dev split and let Optuna maximize the score.  
  - After HPO, we reset to the supervised snapshot, apply best params, and run full RL for `--rl_epochs` on the full train set.  

---

### Evaluation & Model Selection
- **Metric**: dev penalized BLEU (SacreBLEU, `effective_order=True`).  
- **Consistency**: same BLEU flavor for reward (sentence-level) and reporting (corpus-level) to reduce metric drift.  
- **Checkpointing**: save `best_bart_rl_model.pt` whenever dev penalized BLEU improves during RL; reload the best at the end.  

---

### Relation to Prior Work & Our Contribution
- **Li et al., 2018 — Paraphrase Generation with Deep RL.**  
  They train a generator plus a learned evaluator and fine-tune with the evaluator’s signal.  
  We simplify this by using SCST with a penalized BLEU reward—no extra evaluator to train—yielding a lean, reproducible pipeline that still optimizes a task-aligned objective.  

- **Bandel et al., 2022 — Quality-Guided Paraphrase Generation.**  
  They introduce explicit control along semantic/syntactic/lexical axes.  
  We adopt this controllability idea with a lightweight reward blend (BLEU proxy, length-variation proxy, lexical-overlap proxy) selectable via `--quality_weights`, integrating seamlessly into SCST without additional models.  

**Net contribution:**  
A practical, stable paraphrase system that:  
1. Aligns training with evaluation via penalized BLEU.  
2. Stabilizes RL with MLE mixing and variance guards.  
3. Offers optional controllability with a simple, interpretable reward blend.  
4. Includes reproducible HPO for RL-sensitive knobs.  

---

### Limitations & Design Trade-offs
- Sentence-level BLEU is a proxy for semantics; excessive exploration can still drift meaning.  
- Penalized BLEU can oscillate across RL epochs; early stopping and best-checkpoint selection mitigate this.  
- The quality-guided proxies are lightweight (not learned quality models); they trade precision for simplicity and speed.  

---

### Where to Find Things in Code
- **Data & collation**: `transform_data`, `ParaphraseRLDataset`, `rl_transform_data`  
- **Decoding**: `greedy_decode`, `sample_decode`  
- **Rewards**: `penalized_bleu_per_sample`, `quality_guided_reward`  
- **PG log-probs**: `sequence_logprob`  
- **Training loops**: `train_model` (MLE), `rl_finetune_epoch` (SCST)  
- **Orchestration & HPO**: `finetune_paraphrase_generation` (flags: `--use_rl`, `--quality_weights`, `--hpo`, `--hpo_*`, `--rl_*`)  



# Experiments

In this section, we describe the key experiments executed during the development of our paraphrase generation model using reinforcement learning (RL) and Self-Critical Sequence Training (SCST). The experiments aim to evaluate the performance of different models and hyperparameter configurations.

#### 1. **Baseline Model: BART with Supervised Learning (MLE)**

**Task**: Paraphrase Generation  
**Model**: BART (Pre-trained)  
**Method**: Supervised training (Maximum Likelihood Estimation - MLE)  
**Hyperparameters**:  
- Epochs: 7  
- Batch Size: 8  
- Learning Rate: 1e-5  

**Expectations**:  
The baseline model is expected to generate paraphrases but will be limited by the standard MLE training, hence it is expected to copy the output from the input sentence, resulting in a low penalized BLEU score.

**Initial Improvement**:  
Initially, we performed some tweaking to the `transform` function for the baseline model. Specifically, we implemented **padding masking** where target pads are ignored during the loss computation by setting `labels[labels == pad_id] = -100`. This ensures that tokens that are padding do not contribute to the cross-entropy loss, preventing noise in the training process. 

**Changes Compared to Previous Model**:  
- Introduced padding masking to ensure that padded tokens are excluded from loss calculations, thus improving training efficiency and reducing noise.

**Results**:  
After this improvement in the baseline model, we observed a significant performance increase, achieving a **26.21 penalized BLEU score** on the development dataset, which reflects a reasonable starting point for paraphrase generation.

| Epoch | Loss     | Average Loss | BLEU Score | Negative BLEU Score | Penalized BLEU | Dev BLEU Score |
|-------|----------|--------------|------------|---------------------|----------------|----------------|
| 1     | 1.4742   | 1.5799       | 48.06      | 2.71                | 2.51           | 2.51           |
| 2     | 1.0772   | 1.1527       | 46.93      | 8.49                | 7.66           | 7.66           |
| 3     | 1.1725   | 0.9554       | 46.98      | 10.80               | 9.76           | 9.76           |
| 4     | 0.8185   | 0.7456       | 45.95      | 19.08               | 16.85          | 16.85          |
| 5     | 0.4700   | 0.5236       | 42.95      | 27.81               | 22.97          | 22.97          |
| 6     | 0.3281   | 0.3323       | 43.01      | 29.48               | 24.38          | 24.38          |
| 7     | 0.2525   | 0.2159       | 42.29      | 32.23               | 26.21          | 26.21          |


#### 2. **SCST with Penalized BLEU Reward**

**Task**: Paraphrase Generation  
**Model**: BART + SCST (Self-Critical Sequence Training)  
**Method**: Reinforcement Learning with Penalized BLEU Reward  
**Hyperparameters**:  
- RL Epochs: 12  
- Learning Rate: 2e-6  
- Lambda RL: 0.3 (mixing with MLE)  
- Batch Size: 8  

**Expectations**:  
We expect SCST to improve the model's ability to generate paraphrases by aligning training directly with the evaluation metric (penalized BLEU). SCST should help the model explore a wider range of paraphrases and discourage overfitting to the training data. By directly optimizing penalized BLEU, the model can focus on generating more diverse and faithful paraphrases.

**Changes Compared to Previous Model**:  
- Introduction of reinforcement learning for fine-tuning after initial MLE training.
- Reward function: Penalized BLEU score with direct feedback from a comparison between greedy decoding and sampled outputs. This incentivizes better exploration of paraphrasing possibilities.
- Introduced **smoothing by BLEU** (`effective_order=True`) to reduce length and low-count artifacts in BLEU score computation, improving the robustness of evaluation.

**Evaluation Metrics**:  
- **Dev Penalized BLEU**  
- **Sampled vs. Greedy Rewards** (Evaluation of exploration vs. exploitation)

#### STSC-Focused RL Training (12 Epochs)

| RL Epoch | BLEU (ref→hyp) | 100 − BLEU (input→hyp) | Penalized BLEU |
|---|---:|---:|---:|
| 1 | 39.04 | 38.15 | 28.65 |
| 2 | 42.21 | 30.65 | 24.88 |
| 3 | 38.13 | 40.46 | 29.67 |
| 4 | 38.08 | 40.94 | 29.98 |
| 5 | 31.37 | 50.51 | 30.47 |
| 6 | 34.57 | 44.89 | 29.85 |
| 7 | 42.45 | 29.67 | 24.22 |
| 8 | 38.42 | 39.52 | 29.20 |
| 9 | 39.36 | 38.25 | 28.95 |
| 10 | 35.02 | 47.18 | 31.77 |
| 11 | 37.99 | 39.32 | 28.72 |
| 12 | 32.84 | 50.13 | 31.66 |

- **Best Penalized BLEU:** 31.77 (Epoch 10)  
- **Improvement:** 26.21 → 31.77 from start of RL phase  
- **Observation:** Penalized BLEU fluctuates due to trade-off between fidelity to input and semantic divergence.


#### 3. **Quality-Guided Reward with Optional Weights**

**Task**: Paraphrase Generation  
**Model**: BART + SCST with Quality-Guided Reward  
**Method**: Reinforcement Learning with Quality-Guided Reward (Semantic, Syntactic, Lexical)  
**Hyperparameters**:  
- **Quality Weights**: "sem syn lex" (set via command-line arguments)  
- **RL Epochs**: 12  
- **Learning Rate**: 2e-6  
- **Lambda RL**: 0.3  
- **Batch Size**: 8  

**Expectations**:  
By introducing explicit control over semantic, syntactic, and lexical variations in the paraphrase generation, we anticipate the model will produce more diverse and semantically faithful paraphrases. This improvement is expected to address areas where the standard SCST-based model might struggle, particularly in maintaining a balance between adequacy and diversity.

**Changes Compared to Previous Model**:  
- **Additional control** for generation quality via user-defined weights on semantic, syntactic, and lexical factors.
- This allows the user to fine-tune the importance of different aspects of paraphrase generation, providing a more flexible approach to handling quality.

**Evaluation Metrics**:  
- **Dev Penalized BLEU**  
- **Quality Evaluation** (Semantic, Syntactic, Lexical)

#### 4. **Hyperparameter Optimization (HPO) with Optuna**

**Task**: Hyperparameter Tuning for SCST  
**Method**: Hyperparameter Optimization using Optuna with TPE (Tree-structured Parzen Estimator)  
**Hyperparameters to Optimize**:  
- **rl_lr** (learning rate for RL)  
- **lambda_rl** (weight between RL and MLE)  
- **rl_max_len** (maximum sequence length)  
- **rl_batch_size** (batch size)  

**Expectations**:  
The objective of this experiment is to find the optimal combination of hyperparameters that maximizes the penalized BLEU score on the development set. By doing so, we aim to improve model performance and training efficiency by identifying the best settings for reinforcement learning-related hyperparameters.

**Changes Compared to Previous Model**:  
- Introduction of an **automatic hyperparameter search** with Optuna, enabling the identification of the best RL-related hyperparameters without requiring manual tuning.
- This process aims to streamline the search for the most effective settings for RL fine-tuning.

**Evaluation Metrics**:  
- **Dev Penalized BLEU**  
- **Model performance** based on the best hyperparameters identified through the HPO process.

  **Results**:
- For each trial, we run short RL (`--hpo_rl_epochs`, 2)
  ### Hyperparameter Optimization Results

### Hyperparameter Optimization Results (Top 5 Trials Based on Highest Penalized BLEU)

| Trial | RL Learning Rate (rl_lr) | Lambda RL (lambda_rl) | Max Length (rl_max_len) | Batch Size (rl_batch_size) | Penalized BLEU Score |
|-------|--------------------------|------------------------|--------------------------|---------------------------|----------------------|
| 0     | 2.88e-06                 | 0.3587                 | 80                       | 4                         | 30.95               |
| 10    | 6.67e-06                 | 0.2913                 | 40                       | 4                         | 29.03               |
| 16    | 1.27e-06                 | 0.3211                 | 80                       | 16                        | 28.55               |
| 24    | 4.11e-06                 | 0.3640                 | 40                       | 4                         | 28.42               |
| 20    | 1.19e-06                 | 0.4693                 | 60                       | 4                         | 28.62               |

### Best Hyperparameters:
- **Best Trial**: Trial 0
- **Parameters**: 
  - **rl_lr**: 2.88e-06
  - **lambda_rl**: 0.3587
  - **rl_max_len**: 80
  - **rl_batch_size**: 4
  - **Penalized BLEU Score**: 30.95
    
- It can be interpreted from that table that batch size = 4 and learning rate in e-06 gives better results.



### Best Hyperparameters:
- **Best Trial**: Trial 0
- **Parameters**: 
  - **rl_lr**: 2.88e-06
  - **lambda_rl**: 0.3587
  - **rl_max_len**: 80
  - **rl_batch_size**: 4
- **Best Penalized BLEU Score**: 30.95

<img width="1234" height="608" alt="image" src="https://github.com/user-attachments/assets/0979a4a2-4756-4cdd-a37c-80323bcee2d5" />

#### Two-Phase RL Training with Weighted Quality and STSC

We implemented a two-phase reinforcement learning (RL) training strategy to further optimize paraphrase generation:

1. **Phase 1 – Quality-Focused RL:**  
   - Trained for **10 epochs** using weighted rewards `[0.7, 0.2, 0.1]` corresponding to **semantic similarity, syntactic variation, and lexical variation**.  
   - This phase prioritized high-quality paraphrase generation while maintaining syntactic and lexical diversity.

2. **Phase 2 – STSC-Focused RL:**  
   - Loaded the resulting weights from Phase 1 into the model and performed **10 more RL epochs** targeting **STSC (sentence-to-sentence consistency)**.  
   - All epochs in both phases used **optimized hyperparameters** identified through prior HPO experiments.

**Results – Penalized BLEU:**

| Phase/Epoch | BLEU (ref→hyp) | 100 − BLEU (input→hyp) | Penalized BLEU |
|---|---:|---:|---:|
| RL Epoch 1 | 43.756 | 29.713 | 25.002 |
| RL Epoch 2 | 40.295 | 39.688 | 30.754 |
| RL Epoch 3 | 26.519 | 59.586 | 30.388 |
| *RL Epoch 4 (best)* | *35.857* | *48.581* | **33.500** |
| RL Epoch 5 | 35.669 | 48.321 | 33.146 |
| RL Epoch 6 | 35.994 | 47.646 | 32.980 |

**Observations and Possible Reasons for Improvement:**  
- **Better penalized BLEU:** The two-phase approach led to a **peak penalized BLEU of 33.5**, higher than any previous single-phase RL run.  
- **Quality weighting in Phase 1** allowed the model to internalize semantic fidelity while maintaining syntactic and lexical variety.  
- **Phase 2 STSC fine-tuning** ensured stronger sentence-to-sentence consistency, improving the penalization term and stabilizing BLEU scores.  
- **Cumulative effect:** Starting STSC training from weights already optimized for quality helped the model converge faster and more effectively than training from scratch.  
- Overall, **progressive optimization with phased objectives and tuned hyperparameters** produced a model that balanced creativity and accuracy, yielding higher penalized BLEU.










## Results 

| **Paraphrase Type Generation (PTG)**                        | **Penalized BLEU Score** |
|-------------------------------------------------------------|--------------------------|
| Baseline with padding masking                               | 26.21                    |
| SCST with Penalized BLEU Reward                              | 31.77                    |
| Quality-Guided Reward with Optional Weights                  | 52.11%                   |
| Two-Phase RL Training with Weighted Quality and STSC        | 33.50                    |

Discuss your results, observations, correlations, etc.

Results should have three-digit precision.
 

### Hyperparameter Optimization 
### Supervised Learning:
For the baseline model, we experimented with different learning rates and epochs. Through these trials, we observed that a learning rate of **1e-5** resulted in the highest penalized BLEU score within just **7 epochs**. Beyond this point, the loss decreased at a very slow rate, and the penalized BLEU score either stagnated or showed only minimal improvement. On the other hand, using smaller learning rates required more epochs to reach the optimal score, but the training time increased significantly. This insight helped us settle on **1e-5** for the learning rate and **7 epochs** for training.

#### Batch Size:
We selected a **batch size of 8** as it provided a good balance between computational efficiency and model performance. A batch size smaller than 8 led to more variance in the gradient updates, while a larger batch size resulted in higher memory consumption and slower training times. Batch size 8 yielded stable results without sacrificing too much on computational efficiency.

### Hyperparameter Optimization (HPO) for SCST:
For the **SCST** model, some hyperparameters were selected through **Optuna** for automatic tuning, specifically focusing on **rl_lr**, **lambda_rl**, **rl_max_len**, and **rl_batch_size**. The choice of these parameters was made to maximize the penalized BLEU score, which is the task-specific reward.

### Quality-Guided Reward:
The weights for **semantic**, **syntactic**, and **lexical** components were selected based on the empirical results and the trade-off we wanted to achieve between **diversity** and **semantic accuracy**. We experimented with different weight combinations and found that a balanced combination of **0.7** for **semantic similarity**, **0.2** for **syntactic variation**, and **0.1** for **lexical variation** resulted in the most diverse yet semantically coherent paraphrases. This allowed for better control over the model’s output while aligning with the task’s objectives.


## Visualizations 
<img width="845" height="574" alt="image" src="https://github.com/user-attachments/assets/5bb233bd-77c9-4300-ac88-9eed95583079" />
<img width="845" height="574" alt="image" src="https://github.com/user-attachments/assets/96f75088-9f53-45c5-8e6b-f2cecb8a7125" />


Add relevant graphs of your experiments here. Those graphs should show relevant metrics (accuracy, validation loss, etc.) during the training. Compare the  different training processes of your improvements in those graphs. 

For example, you could analyze different questions with those plots like: 
- Does improvement A converge faster during training than improvement B? 
- Does Improvement B converge slower but perform better in the end? 
- etc...


