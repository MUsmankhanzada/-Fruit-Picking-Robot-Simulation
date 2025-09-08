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
## Problem & Objective

We tackle **paraphrase generation** with the goal of producing fluent outputs that **preserve meaning** while **discouraging copying** from the source. Training is aligned with evaluation using a **penalized BLEU** objective, used both as the RL reward and for dev model selection:

$$
R(\text{ref}, \text{inp}, \text{hyp}) = \text{BLEU}(\text{ref}, \text{hyp}) \times \frac{100 - \text{BLEU}(\text{inp}, \text{hyp})}{52}
$$

We compute BLEU with **SacreBLEU** (`effective_order=True`) to mitigate length and low-count artifacts.
```ini
encoder_input = sentence1  <SEP>  sentence1_segment_location  <SEP>  paraphrase_type_ids
```
where <SEP> is the tokenizer’s sep_token (or </s> fallback).

**Targets:** `sentence2` (gold paraphrase).
**Padding masking:** target pads are ignored via labels `[labels == pad_id] = -100`, ensuring <pad> tokens do not contribute to the cross-entropy, otherwise there will be noise in the loss.
### Model & Data Encoding

**Backbone:** `facebook/bart-large` (seq2seq)

**Inputs:** each example is serialized as






# Experiments
Keep track of your experiments here. What are the experiments? Which tasks and models are you considering?

Write down all the main experiments and results you did, even if they didn't yield an improved performance. Bad results are also results. The main findings/trends should be discussed properly. Why a specific model was better/worse than the other?

You are **required** to implement one baseline and improvement per task. Of course, you can include more experiments/improvements and discuss them. 

You are free to include other metrics in your evaluation to have a more complete discussion.

Be creative and ambitious.

For each experiment answer briefly the questions:

- What experiments are you executing? Don't forget to tell how you are evaluating things.
- What were your expectations for this experiment?
- What have you changed compared to the base model (or to previous experiments, if you run experiments on top of each other)?
- What were the results?
- Add relevant metrics and plots that describe the outcome of the experiment well. 
- Discuss the results. Why did improvement _A_ perform better/worse compared to other improvements? Did the outcome match your expectations? Can you recognize any trends or patterns?

## Results 
Summarize all the results of your experiments in tables:

| **Stanford Sentiment Treebank (SST)** | **Metric 1** |**Metric n** |
|----------------|-----------|------- |
|Baseline |45.23%           |...            | 
|Improvement 1          |58.56%            |...          
|Improvement 2        |52.11%|...|
|...        |...|...|

| **Quora Question Pairs (QQP)** | **Metric 1** |**Metric n** |
|----------------|-----------|------- |
|Baseline |45.23%           |...            | 
|Improvement 1          |58.56%            |...          
|Improvement 2        |52.11%|...|
|...        |...|...|

| **Semantic Textual Similarity (STS)** | **Metric 1** |**Metric n** |
|----------------|-----------|------- |
|Baseline |45.23%           |...            | 
|Improvement 1          |58.56%            |...          
|Improvement 2        |52.11%|...|
|...        |...|...|

| **Paraphrase Type Detection (PTD)** | **Metric 1** |**Metric n** |
|----------------|-----------|------- |
|Baseline |45.23%           |...            | 
|Improvement 1          |58.56%            |...          
|Improvement 2        |52.11%|...|
|...        |...|...|

| **Paraphrase Type Generation (PTG)** | **Metric 1** |**Metric n** |
|----------------|-----------|------- |
|Baseline |45.23%           |...            | 
|Improvement 1          |58.56%            |...          
|Improvement 2        |52.11%|...|
|...        |...|...|

Discuss your results, observations, correlations, etc.

Results should have three-digit precision.
 

### Hyperparameter Optimization 
Describe briefly how you found your optimal hyperparameter. If you focussed strongly on Hyperparameter Optimization, you can also include it in the Experiment section. 

_Note: Random parameter optimization with no motivation/discussion is not interesting and will be graded accordingly_

## Visualizations 
Add relevant graphs of your experiments here. Those graphs should show relevant metrics (accuracy, validation loss, etc.) during the training. Compare the  different training processes of your improvements in those graphs. 

For example, you could analyze different questions with those plots like: 
- Does improvement A converge faster during training than improvement B? 
- Does Improvement B converge slower but perform better in the end? 
- etc...

## Members Contribution 

**Usman Khanzada:** Implemented the paraphrase generation baseline task using BART and then introduced improvements to enhance its performance, also managed group organization

**Member 2:**

**Member 2:**

**Member 2:**

**Member 2:**
 

# AI-Usage Card
Artificial Intelligence (AI) aided the development of this project. Please add a link to your AI-Usage card [here](https://ai-cards.org/).
        
