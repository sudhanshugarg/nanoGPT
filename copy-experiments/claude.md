# Copy Task Experiments

This folder tracks hyperparameter experiments and results for the copy task to identify optimal configurations.

## How to Record Experiments

For each experiment run:

1. Create a dated entry below with the format: `## Experiment - YYYY-MM-DD - [Description]`
2. Document the hyperparameters used
3. Record the training configuration
4. Note the results (loss, accuracy, observations)
5. Add any key findings or insights

## Experiment Template

```
## Experiment - YYYY-MM-DD - [Brief description of what was tested]

**Hyperparameters:**
- batch_size: X
- learning_rate: X
- num_layers: X
- embed_dim: X
- max_steps: X
- [add others as needed]

**Config:**
- Dataset: copy
- Model: [model details]
- [other relevant config]

**Results:**
- Final train loss: X
- Final eval loss: X
- Best eval loss: X (at step Y)
- Training time: X minutes
- Key observations: [notes]

**Conclusion:**
[What worked, what didn't, next steps]
```

## Experiments

[See results.md for experiment records]
