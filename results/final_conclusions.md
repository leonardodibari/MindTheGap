# Final conclusions

- **Best for interpolation:** `Ensemble (w=0.46)` has the lowest random-test MAE (0.01313).
- **Best for unseen scaffolds:** `Ensemble (w=0.46)` has the lowest scaffold-test MAE (0.01968).
- **Explicit geometry:** SchNet changes MAE by -0.00031 on random test and +0.00187 on scaffold test; its benefit is therefore concentrated where the positive difference is larger.
- **Error complementarity:** residual Pearson/Spearman correlations are 0.999/0.496 on random test and 1.000/0.573 on scaffold test. Pearson is dominated by shared extreme-target errors, while rank correlation better reflects typical-molecule complementarity.
- **Ensemble robustness:** validation selected `w=0.46` for SchNet. Relative to the better individual model, ensemble MAE improves by +0.00113 on random test and +0.00027 on scaffold test.

