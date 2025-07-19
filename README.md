Cascade Cosmology: A Dark Energy-Free Model Driven by Phase Transitions
📜 Brief Description
This project presents the development and testing of a novel cosmological model, dubbed PLM-FP (Phenomenological Linear Model with Phase Transitions). It offers an alternative framework for understanding the evolution of the universe without the need for dark energy or a cosmological constant (Λ). The core hypothesis is that the observed effects attributed to dark energy are a consequence of the dynamic nature of physical time, whose tempo evolves depending on the energy density of the medium.
The model posits that structure formation (the formation of galaxies and other domains) actively alters the properties of spacetime, leading to observable consequences that differ from those of the standard ΛCDM model.
🛠️ Technologies and Data
Language: Python 3
Core Libraries: NumPy, SciPy, Matplotlib, Astropy
MCMC Analysis: emcee for Markov Chain Monte Carlo, corner for visualization.
CMB Analysis: CLASS (Cosmic Linear Anisotropy Solving System) for theoretical power spectrum calculations.
Data Sets:
Type Ia Supernovae: Pantheon+ (1701 data points)
Baryon Acoustic Oscillations (BAO): A compilation of H(z) and d_A(z) measurements.
Cosmic Microwave Background (CMB): Reference values and power spectra from the Planck 2018 data release.
📊 Summary of Results
When testing the PLM-FP model against standard cosmological datasets, we found several key results:
Superior Data Fit: The PLM-FP model demonstrates a statistically significant better fit to late-universe data (SN+BAO) compared to the standard ΛCDM model.
Consistency with the Early Universe: The model, with parameters determined from late-universe data, successfully reproduces the main features of the CMB power spectrum. The predicted spectrum is closer to observations than that of a similarly constrained ΛCDM model.
Explanation for Cosmological Tensions: The model offers a new perspective on the Hubble Tension, re-framing it not as a contradiction but as a predicted consequence of the difference between the global and local dynamics of time.
Specific Predictions: The theory leads to specific, falsifiable predictions, such as a global Hubble parameter value of H₀ ≈ 47 km/s/Mpc and the existence of a local blueshift effect of z_local ≈ -0.05.
For a more detailed scientific narrative of the model's evolution and results, please see the README file in the mcmc_analysis/ directory.
🚀 Installation and Usage
The project was developed and tested under Windows 11 with WSL2 (Ubuntu), using Python 3.11 (recommended) in a conda environment.
1. Environment Setup
Generated bash
# 1. Clone the repository
git clone https://github.com/aaamil13/PLM-FP.git
cd PLM-FP

# 2. Create and activate a conda environment
conda create -n cosmo_env python=3.11 -y
conda activate cosmo_env

# 3. Install core dependencies
pip install numpy scipy matplotlib emcee corner h5py psutil astropy

# 4. (Optional) Install CLASS and Planck Likelihood

# Required only for advanced CMB spectrum tests.
pip install classy planck-lite-py

# Note: You may need to compile CLASS from source for full functionality.
Use code with caution.
Bash
2. Reproducing Key Results
All scripts should be run from the project's root directory (Test_5/).
A) Run the Final MCMC Simulation:
This script runs the full MCMC analysis for the PLM model with its most successful configuration (with a CMB prior). Warning: This process is computationally intensive and may take hours or days. You can skip this step by using the provided .h5 results file.
Generated bash

# Runs the CMB-constrained simulation
python mcmc_analysis/runners/run_mcmc.py --model PLM
Use code with caution.
Bash
B) Statistical Model Comparison (Most Important Result):
This script uses the MCMC output (the PLM_CMB_constrained_optimized_checkpoint.h5 file) to quantitatively compare the PLM and ΛCDM models.
Generated bash

# Ensure compare_models.py is configured to read the correct .h5 file
python mcmc_analysis/runners/compare_models.py
Use code with caution.
Bash
C) Generate Analysis Plots:
These scripts use the MCMC results to generate key visualizations.
Generated bash
# Generate the Hubble diagram residuals plot
python mcmc_analysis/runners/plot_hubble_residuals.py

# Generate the plot for the local evolution of H₀
python mcmc_analysis/runners/test_h0_evolution.py

# Generate the CMB spectrum plot (requires CLASS)

# First, generate the w(z) file
python mcmc_analysis/runners/generate_effective_w.py

# Then, run the script that calls CLASS and plots the result
python mcmc_analysis/runners/run_class_and_plot.py
Use code with caution.
Bash

# 🤝 Collaboration and Methodology
This project is the result of an innovative collaboration between a human researcher and several AI models. The physical intuition, core hypotheses, and critical analysis were human-led, while the AI assistants facilitated rapid prototyping, code implementation, debugging, and the structuring of results. This workflow demonstrates the potential of AI-augmented science to accelerate the research cycle.

# 📜 License
This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).
![cc-by-nc-sa-shield](https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png)
You are free to share and adapt this material for non-commercial purposes, provided you give appropriate credit, share alike, and indicate if changes were made.
