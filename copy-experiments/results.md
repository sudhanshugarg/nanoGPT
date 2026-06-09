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

---

## Experiment 2 - 2026-06-09 - Larger batch and block size

**Hyperparameters:**
- batch_size: 512
- learning_rate: 1e-4
- n_layer: 2
- n_head: 1
- n_embd: 384
- dropout: 0.2
- block_size: 128
- max_iters: 15000
- warmup_iters: 100
- beta2: 0.99

**Config:**
- Dataset: copy
- Device: GPU (default, not cpu)
- Model: Baby GPT (2 layers, 1 head, 384 embedding dim)
- gradient_accumulation_steps: 1
- eval_interval: 500
- eval_iters: 200
- compile: False

**Results:**
- Final train loss: 0.0000 (at step 15000)
- Final val loss: 0.6355 (at step 15000)
- Best val loss: 0.1494 (at step 3000)
- Output directory: out-copy7

**Conclusion:**
Severe overfitting with larger batch size (512) and block size (128). Training loss drops to near zero but validation loss diverges after iteration 3000, reaching 0.6355 by the end. The model completely fails to generalize - much worse than Experiment 1. The increased batch size and context length appear to hurt generalization significantly.
