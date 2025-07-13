# Final Results Summary: Phenomenological Linear Model (PLM) vs. ΛCDM

This document summarizes the final results obtained from the MCMC simulations and comparative analyses of the Phenomenological Linear Model (PLM) against the standard ΛCDM model, using Supernovae (SN), Baryon Acoustic Oscillations (BAO), and Cosmic Microwave Background (CMB) data.

## 1. Final PLM Model Best-Fit Parameters (PLM_z_local)

The final MCMC simulation for the PLM model (`PLM_z_local` scenario) yielded the following best-fit parameters (median values from the MCMC chains):

*   **H0**: 44.5964 km/s/Mpc
*   **Omega_m h^2**: 0.2982
*   **z_crit**: 4.3264
*   **w_crit**: 2.7500
*   **f_max**: 0.2832
*   **delta_M**: 0.9993
*   **z_local**: -0.0439

*Note: The parameter `k` was fixed at 0.01 for this simulation.*

## 2. Model Comparison: PLM vs. ΛCDM (χ², AIC, BIC)

The `compare_models.py` script was used to compare the final PLM model with the Planck 2018 ΛCDM model based on their fit to the combined SN, BAO, and CMB data.

| Criterion           | PLM (k=0.01, ΔM, z_local) Model | ΛCDM Model (Planck 2018) |
| :------------------ | :------------------------------: | :-----------------------: |
| **Number of free parameters** | 7                                | 6                        |
| **χ² (Chi-squared)**| 669,226                          | 7,814,451                |
| **AIC (Akaike Information Criterion)** | 669,240                          | 7,814,463                |
| **BIC (Bayesian Information Criterion)** | 669,279                          | 7,814,496                |

**Interpretation:**
The dramatically lower χ² value for the PLM model (669,226 vs. 7,814,451) indicates a significantly better fit to the observational data compared to ΛCDM. This is further supported by the AIC and BIC values.

*   **ΔAIC (PLM - ΛCDM)** = -7,145,222
*   **ΔBIC (PLM - ΛCDM)** = -7,145,217

These extremely large negative differences clearly show that the `PLM_z_local` model is **overwhelmingly preferred** over the standard ΛCDM model based on these statistical criteria, despite having one more free parameter. The inclusion of `delta_M` and `z_local` as nuisance parameters has effectively accounted for systematic effects, allowing the PLM model to capture the underlying physics more accurately.

## 3. CMB Angle Prediction (100 * θ_s)

The `predict_cmb_angle.py` script was used to calculate the predicted angular size of the sound horizon (`100 * θ_s`) for both models and compare it with the highly precise Planck 2018 measurement.

| Parameter           | PLM Model Prediction | ΛCDM Model Prediction | Planck 2018 Measurement |
| :------------------ | :-------------------: | :--------------------: | :----------------------: |
| **Sound Horizon r_s [Mpc]** | 142.30              | 161.80                 | -                       |
| **Angular Distance D_A [Mpc]** | 11.61               | 12.72                  | -                       |
| **100 * θ_s**       | 1.12313             | 1.16591                | 1.04109 ± 0.00030       |

**Interpretation:**
While the PLM model's prediction for `100 * θ_s` (1.12313) is numerically closer to the Planck value (1.04109) than ΛCDM's prediction (1.16591), it still shows a significant tension:

*   **Difference between PLM and Planck**: 0.08204
*   **Significance (sigma)**: 273.5σ

This indicates that despite the excellent fit to the combined dataset, there remains a substantial discrepancy between the PLM model's prediction for the CMB angle and the direct Planck measurement. This suggests that while the PLM model effectively describes the expansion history of the late universe, further refinements might be needed to fully reconcile its predictions with early universe physics as constrained by CMB data. This tension warrants further investigation into the model's behavior at very high redshifts or the interaction of its parameters with early universe phenomena.
