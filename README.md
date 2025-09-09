
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

| **Phase/Epoch** | **BLEU (ref→hyp)** | **100 − BLEU (input→hyp)** | **Penalized BLEU** |
|-----------------|--------------------|----------------------------|--------------------|
| RL Epoch 1      | 42.36              | 30.73                      | 25.04              |
| RL Epoch 2      | 25.59              | 58.04                      | 28.56              |
| RL Epoch 3      | 39.67              | 36.20                      | 27.61              |
| RL Epoch 4      | 42.18              | 30.55                      | 24.78              |
| RL Epoch 5      | 35.69              | 39.17                      | 26.88              |
| RL Epoch 6      | 35.31              | 41.57                      | 28.23              |
| RL Epoch 7      | 41.47              | 32.24                      | 25.71              |
| RL Epoch 8      | 36.71              | 42.05                      | 29.69              |
| RL Epoch 9      | 38.39              | 39.22                      | 28.96              |
| RL Epoch 10     | 41.38              | 31.11                      | 24.75              |
| RL Epoch 11     | 34.07              | 43.58                      | 28.56              |
| RL Epoch 12     | 43.67              | 24.97                      | 20.97              |

**RL Improved BLEU**: 26.21 → 29.69


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


### Quality-Guided Reward with Optional Weights vs SCST with Penalized BLEU Reward

As expected, **SCST with Penalized BLEU Reward** performs better in terms of **Penalized BLEU** because it is specifically optimized for this metric. The SCST method directly maximizes **Penalized BLEU**, which leads to higher diversity in the paraphrases, as evidenced by the improvement from **26.21** (baseline) to **31.77**. This method focuses on reducing input similarity while ensuring that the generated output closely matches the reference.

On the other hand, **Quality-Guided Reward with Optional Weights** focuses more on enhancing **semantic similarity** by balancing various factors like **semantic**, **syntactic**, and **lexical** variations. While it achieved a **Penalized BLEU** score of **29.69**, which is lower than SCST's **31.77**, it improved the **BLEU (ref→hyp)** score to **36.71**. However, this focus on quality and balance led to higher **100 − BLEU (input→hyp)** (42.05), suggesting that the model prioritized improving reference similarity at the expense of input diversity.

In summary, **SCST** is tailored to improve **Penalized BLEU**, making it more effective for tasks requiring diversity and reduced input similarity. **Quality-Guided Reward**, on the other hand, emphasizes semantic quality, making it better suited for tasks where reference alignment is critical, albeit with a trade-off in diversity.
  

  ## Key Takeaways

### 1. **Change in Numbers During Generation**  
One notable observation during the evaluation of the paraphrase generation was the change in numbers within the generated sentences. Specifically, the numerical values in the original input sentence were altered in the paraphrases, even when they were expected to remain unchanged. This issue likely stemmed from the lack of high-quality paraphrasing in the model output. We hypothesize that the model’s inability to correctly retain key numerical information and other sensitive tokens might have been caused by suboptimal quality in the generated paraphrases, which are unable to preserve such specific details.

### 2. **Trade-off Between Penalized BLEU and BLEU Score**  
We noticed a consistent trend where an increase in the **Penalized BLEU score** was accompanied by a decrease in the **BLEU score**. This suggests that there exists a trade-off between these two metrics. The **BLEU score** is a measure of similarity to the reference, but penalized BLEU tries to mitigate issues where the generated text overly matches the input (i.e., it discourages repetition of the input text). Thus, models that generate highly similar paraphrases to the reference might achieve high BLEU scores but lower penalized BLEU scores, and vice versa.

### 3. **Trade-off Between Quality-Generated and Penalized BLEU Score**  
A similar trade-off was observed between the **Quality-Generated Paraphrases** and the **Penalized BLEU Score**. In our experiments, as the **Quality-Generated** metric improved (using weighted rewards for semantic, syntactic, and lexical variations), the **Penalized BLEU** score also showed improvement, but the trade-off with BLEU was still evident. A model focusing heavily on quality might not always optimize for **BLEU**, as it sacrifices some level of similarity with the reference sentence in favor of generating more semantically diverse and syntactically varied output.

### 4. **Optimal Balance Between BLEU and Penalized BLEU**  
The experiments indicate that the best-performing models, in terms of **Penalized BLEU**, were able to strike a balance between maintaining a high similarity to the reference sentence while introducing enough variation to avoid the pitfalls of overly repetitive text generation. This balance is crucial for improving both the **BLEU** and **Penalized BLEU** scores simultaneously, making sure that paraphrases are both semantically correct and sufficiently distinct from the input text.




# Results 

| **Paraphrase Type Generation (PTG)**                        | **Penalized BLEU Score** | **BLEU (ref→hyp)** | **100 − BLEU (input→hyp)** |
|-------------------------------------------------------------|--------------------------|--------------------|----------------------------|
| Baseline with padding masking                               | 26.21                    | 42.29              | 32.23                      |
| SCST with Penalized BLEU Reward                              | 31.77                    | 35.02              | 47.18                      |
| Quality-Guided Reward with Optional Weights                  | 29.69                   |  36.71             |  42.05                      |
| Two-Phase RL Training with Weighted Quality and STSC        | 33.50                    | 35.85              | 48.58                      |

### Results Discussion

The **Paraphrase Type Generation (PTG)** experiments demonstrate how different training methods affect the **Penalized BLEU Score**, **BLEU (ref→hyp)**, and **100 − BLEU (input→hyp)**:

1. **Baseline with Padding Masking (26.21 Penalized BLEU)**:
   - The baseline achieved a **Penalized BLEU** score of **26.21**. The **BLEU (ref→hyp)** of **42.29** and **100 − BLEU (input→hyp)** of **32.23** reflect a moderate balance between reference similarity and input diversity.

2. **SCST with Penalized BLEU Reward (31.77 Penalized BLEU)**:
   - **SCST** improved the **Penalized BLEU** score to **31.77** but resulted in a lower **BLEU (ref→hyp)** of **35.02** and higher **100 − BLEU (input→hyp)** at **47.18**, showing a trade-off between diversity and semantic accuracy.

3. **Quality-Guided Reward with Optional Weights (29.69 Penalized BLEU)**:
   - This method yielded a **Penalized BLEU** score of **29.69**, slightly lower than **SCST**, but with a better **BLEU (ref→hyp)** of **36.71** and a **100 − BLEU (input→hyp)** of **42.05**. It demonstrates that weighted quality aspects can improve the model's semantic alignment but at the cost of diversity.

4. **Two-Phase RL Training with Weighted Quality and STSC (33.50 Penalized BLEU)**:
   - The **Two-Phase RL Training** method produced the highest **Penalized BLEU** score of **33.50**, with a **BLEU (ref→hyp)** of **35.85** and **100 − BLEU (input→hyp)** of **48.58**, striking a good balance between diversity and semantic quality.

### Observations:
**As already discussed in Experiments takeaways**
- **Trade-off Between BLEU and Penalized BLEU**: As **Penalized BLEU** increases, **BLEU (ref→hyp)** tends to decrease, indicating a trade-off between semantic accuracy and diversity.
- **Impact of Rewarding Quality Aspects**: **Quality-Guided Reward** improved reference similarity but led to a slightly lower **Penalized BLEU** than **SCST**.
- **Two-Phase RL Training**: This method, combining broader training with **STSC** fine-tuning, achieved the best balance of diversity and semantic alignment.

### Conclusion:

**Two-Phase RL Training with Weighted Quality and STSC** provides the best overall results, with the highest **Penalized BLEU** score, offering an optimal balance between semantic accuracy and paraphrase diversity.  The key takeaway from this analysis is that paraphrase generation is a complex problem that involves balancing between **retaining important details**, **avoiding repetitive outputs**, and **introducing enough variety** to produce fluent and accurate paraphrases. Achieving the best results requires careful tuning of reward functions, attention to the trade-offs between **BLEU** and **Penalized BLEU** scores, and the integration of quality-guided models that improve semantic richness without sacrificing output accuracy.

 
# Hyperparameter Optimization 
### Supervised Learning:
For the baseline model, we experimented with different learning rates and epochs. Through these trials, we observed that a learning rate of **1e-5** resulted in the highest penalized BLEU score within just **7 epochs**. Beyond this point, the loss decreased at a very slow rate, and the penalized BLEU score either stagnated or showed only minimal improvement. On the other hand, using smaller learning rates required more epochs to reach the optimal score, but the training time increased significantly. This insight helped us settle on **1e-5** for the learning rate and **7 epochs** for training.

#### Batch Size:
We selected a **batch size of 8** as it provided a good balance between computational efficiency and model performance. A batch size smaller than 8 led to more variance in the gradient updates, while a larger batch size resulted in higher memory consumption and slower training times. Batch size 8 yielded stable results without sacrificing too much on computational efficiency.

### Hyperparameter Optimization (HPO) for SCST:
For the **SCST** model, some hyperparameters were selected through **Optuna** for automatic tuning, specifically focusing on **rl_lr**, **lambda_rl**, **rl_max_len**, and **rl_batch_size**. The choice of these parameters was made to maximize the penalized BLEU score, which is the task-specific reward.

### Quality-Guided Reward:
The weights for **semantic**, **syntactic**, and **lexical** components were selected based on the empirical results and the trade-off we wanted to achieve between **diversity** and **semantic accuracy**. We experimented with different weight combinations and found that a balanced combination of **0.7** for **semantic similarity**, **0.2** for **syntactic variation**, and **0.1** for **lexical variation** resulted in the most diverse yet semantically coherent paraphrases. This allowed for better control over the model’s output while aligning with the task’s objectives.


# Visualizations 
## **Baseline with padding masking**
<img width="845" height="574" alt="image" src="https://github.com/user-attachments/assets/5bb233bd-77c9-4300-ac88-9eed95583079" />

### Blue curve (Loss):
- The average loss decreases smoothly and consistently as training progresses, indicating stable convergence.
### Red curve (Penalized BLEU):
- The BLEU score improves steadily as the loss decreases, showing a clear correlation between reduced loss and better performance.
### Observation:
- This phase shows a well-behaved supervised training stage, where the model steadily improves without much fluctuation.

## **RL (SCST) on penalized BLEU**
<img width="845" height="574" alt="image" src="https://github.com/user-attachments/assets/96f75088-9f53-45c5-8e6b-f2cecb8a7125" />

### Blue curve (BLEU):
- The BLEU score fluctuates more significantly compared to the first graph. There are ups and downs across epochs.

### Marked point:
- Epoch 10 is highlighted as the best-performing epoch (~31.8 BLEU).

### Observation:
- Reinforcement Learning (RL) training introduces instability in the optimization process, but it occasionally pushes the model to higher performance peaks compared to supervised training.


## **Quality-Guided Reward with Optional Weights**
<img width="845" height="552" alt="image" src="https://github.com/user-attachments/assets/e39b0b90-7ec3-497a-a5aa-26698865e18a" />

### Key Observations:
- The BLEU score shows moderate fluctuations with no clear upward or downward trend.
- The highest BLEU (~29.7) is at epoch 8.
- The score drops sharply to ~21 by epoch 12, suggesting potential overfitting or model degradation.

### Conclusion:

- **Supervised Training**: The graph of supervised training demonstrates a strong and stable base, with a gradual increase in performance. However, it plateaus at a lower BLEU score, suggesting the model achieves its best performance early on but struggles to improve further.

- **RL (SCST) with Penalized BLEU**: The graph of RL (SCST) training shows more fluctuations and instability in performance. While it achieves higher penalized BLEU scores at certain epochs, the training process is less stable. The model occasionally pushes to higher performance peaks, but the fluctuations suggest a less predictable optimization process.

- **Quality-Guided Reward with Optional Weights**: The graph of quality-guided training indicates a significant improvement in performance over the baseline, yet it exhibits the least stability. While the model performs well in terms of penalized BLEU, it faces larger fluctuations and is the least stable approach. Despite this, RL (SCST) consistently achieves higher penalized BLEU scores, showing it as the more effective strategy for this task.

## References 
- _Paraphrase Generation with Deep Reinforcement Learning [Li, Jiang, Shang et al., 2018]_ https://aclanthology.org/D18-1421/
- _Quality Controlled Paraphrase Generation, [Bandel et al., 2022]_ https://aclanthology.org/2022.acl-long.45/


