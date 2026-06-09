# Copy Task Experiment Results

## Experiment 1 - 2026-06-08 - Initial baseline with default hyperparameters

**Hyperparameters:**
- batch_size: 32
- learning_rate: 1e-4
- n_layer: 2
- n_head: 1
- n_embd: 384
- dropout: 0.2
- block_size: 42
- max_iters: 15000
- warmup_iters: 100
- beta2: 0.99

**Config:**
- Dataset: copy
- Device: cpu
- Model: Baby GPT (2 layers, 1 head, 384 embedding dim)
- gradient_accumulation_steps: 1
- eval_interval: 500
- eval_iters: 200

**Results:**
- Final train loss: 0.0165 (at step 15000)
- Final val loss: 0.1164 (at step 15000)
- Best val loss: 0.1117 (at step 14000)
- Output directory: out-copy4

**Wandb:**
- Project: copy-task
- Run name: copy-sugarg
- URL: https://wandb.ai/sudhanshugarg/copy-task/runs/nhhdmt86

**Conclusion:**
This is the baseline configuration. The model achieves reasonable convergence with the copy task. Best validation loss of 0.1117 at iteration 14000 before slight increase towards the end.
