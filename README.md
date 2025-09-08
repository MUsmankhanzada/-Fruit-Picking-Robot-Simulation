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
  --use_rl --rl_epochs 6 \
  --rl_lr 2e-6 --rl_lambda 0.3 --rl_batch_size 8 --rl_max_len 50 \
  --seed 11711
```


        
