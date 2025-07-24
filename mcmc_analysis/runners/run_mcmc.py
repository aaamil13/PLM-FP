"""
Main script for MCMC simulations
=================================

This script defines log-likelihood functions, priors, and runs
MCMC simulations for the Phase-transition Linear Model (PLM) and the standard ΛCDM model
against Supernovae (SN), BAO, and CMB data.

It uses `logging` for recording progress and errors and `emcee.HDFBackend`
for reliable saving and restoring of the simulation state.
"""

import numpy as np
import emcee
import corner
import sys
import os
import time
import matplotlib.pyplot as plt
import multiprocessing
import argparse
import traceback
import logging
import io
from multiprocessing import Pool
from functools import partial
import psutil


# Disable file locking for HDF5
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# Add project directories to module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mcmc_analysis.models.plm_model_fp import PLM
from mcmc_analysis.models.lcdm_model import LCDM
from mcmc_analysis.likelihoods.sn_likelihood import SupernovaeLikelihood
from mcmc_analysis.likelihoods.bao_likelihood import BAOLikelihood
from mcmc_analysis.likelihoods.cmb_likelihood import CMBLikelihood

from utils.font_config import setup_cyrillic_fonts, clear_font_cache
from utils.encoding_utils import setup_cp1251_environment
from utils.logger_config import setup_cp1251_logger, log_safe


# Global instances - will be initialized in worker processes
sn_likelihood = None
bao_likelihood = None
cmb_likelihood = None
logger = None

# Global CMBLikelihood instance (for use in priors)
cmb_likelihood_instance = CMBLikelihood()

# === Definition of log-probability functions and priors ===

def init_worker():
    """Worker process initialization"""
    global sn_likelihood, bao_likelihood, cmb_likelihood, logger
    
    # Configure logging for the worker process
    logger = setup_cp1251_logger(f'Worker-{os.getpid()}', logging.INFO)
    
    try:
        # Load likelihood data at the start of each worker
        sn_likelihood = SupernovaeLikelihood()
        bao_likelihood = BAOLikelihood()
        cmb_likelihood = CMBLikelihood()
        log_safe(logger, logging.INFO, f"Worker {os.getpid()} initialized successfully")
    except Exception as e:
        log_safe(logger, logging.ERROR, f"Error during worker initialization {os.getpid()}: {e}")
        raise

def log_prior_plm(params):
    """Log-prior function for the PLM model."""
    # Parameters for the new PLM-FP: H0, omega_m_h2, z_crit, w_crit, f_max, k
    H0, omega_m_h2, z_crit, w_crit, f_max, k = params
    if not (40 < H0 < 100 and 0.1 < omega_m_h2 < 0.3 and 0.1 < z_crit < 10.0 and
            0.01 < w_crit < 5.0 and 0.1 < f_max < 0.99 and 0.01 < k < 5.0):
        return -np.inf
    return 0.0

def log_likelihood_plm(params, delta_M=0.0, z_local=0.0):
    """Optimized log-likelihood function for the PLM model."""
    global sn_likelihood, bao_likelihood, cmb_likelihood
    
    try:
        model = PLM(*params)
        
        # Calculate likelihoods in parallel
        log_like_sn = sn_likelihood.log_likelihood(model, delta_M=delta_M, z_local=z_local) # Pass delta_M and z_local here
        log_like_bao = bao_likelihood.log_likelihood(model)
        log_like_cmb = cmb_likelihood.log_likelihood(model)
        
        total_lp = log_like_sn + log_like_bao + log_like_cmb
        
        return total_lp
        
    except Exception as e:
        return -np.inf

def log_probability_plm(params):
    """Overall log-probability function for the PLM model."""
    lp = log_prior_plm(params)
    if not np.isfinite(lp):
        return -np.inf
        
    log_like = log_likelihood_plm(params)
    if not np.isfinite(log_like):
        return -np.inf
        
    return lp + log_like

def log_prior_lcdm(params):
    """Log-prior function for the LCDM model."""
    H0, omega_m_h2, omega_b_h2, n_s, A_s, tau_reio = params
    if not (50 < H0 < 100 and 0.05 < omega_m_h2 < 0.20 and 0.01 < omega_b_h2 < 0.03 and
            0.8 < n_s < 1.2 and 1.0e-9 < A_s < 5.0e-9 and 0.01 < tau_reio < 0.1):
        return -np.inf
    return 0.0

def log_likelihood_lcdm(params):
    """Optimized log-likelihood function for the LCDM model."""
    global sn_likelihood, bao_likelihood, cmb_likelihood
    
    try:
        model = LCDM(*params)
        
        lp = sn_likelihood.log_likelihood(model) + \
             bao_likelihood.log_likelihood(model) + \
             cmb_likelihood.log_likelihood(model)
        return lp
        
    except Exception as e:
        return -np.inf

def log_probability_lcdm(params):
    """Overall log-probability function for the LCDM model."""
    lp = log_prior_lcdm(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_lcdm(params)

# --- Global definitions for PLM with fixed k ---
FIXED_K = 0.01

def log_prior_plm_k_fixed(params):
    H0, omega_m_h2, z_crit, w_crit, f_max = params
    if not (40 < H0 < 100 and 
            0.1 < omega_m_h2 < 0.3 and 
            0.1 < z_crit < 10.0 and
            0.01 < w_crit < 5.0 and 
            0.1 < f_max < 0.99):
        return -np.inf
    return 0.0

def log_probability_plm_k_fixed(params):
    lp = log_prior_plm_k_fixed(params)
    if not np.isfinite(lp):
        return -np.inf
    
    H0, omega_m_h2, z_crit, w_crit, f_max = params
    full_params = [H0, omega_m_h2, z_crit, w_crit, f_max, FIXED_K]
    
    log_like = log_likelihood_plm(full_params) 
    
    if not np.isfinite(log_like):
        return -np.inf
    return lp + log_like
# --- End global definitions ---

# --- Global definitions for PLM with H0 constrained ---
def log_prior_plm_H0_constrained(params):
    H0, omega_m_h2, z_crit, w_crit, f_max = params
    if not (60 < H0 < 70 and           # **Strong prior for H0**
            0.1 < omega_m_h2 < 0.3 and 
            0.1 < z_crit < 10.0 and
            0.01 < w_crit < 5.0 and 
            0.1 < f_max < 0.99):
        return -np.inf
    return 0.0

def log_probability_plm_H0_constrained(params):
    lp = log_prior_plm_H0_constrained(params)
    if not np.isfinite(lp):
        return -np.inf
    
    H0, omega_m_h2, z_crit, w_crit, f_max = params
    full_params = [H0, omega_m_h2, z_crit, w_crit, f_max, FIXED_K] # Use the global FIXED_K
    
    log_like = log_likelihood_plm(full_params) 
    
    if not np.isfinite(log_like):
        return -np.inf
    return lp + log_like
# --- End global definitions ---

# --- Global definitions for PLM final simulation (with delta_M and z_local) ---
def log_prior_plm_final(params):
    H0, omega_m_h2, z_crit, w_crit, f_max, delta_M, z_local = params
    if not (40 < H0 < 100 and             # Wide prior for H0
            0.1 < omega_m_h2 < 0.3 and 
            0.1 < z_crit < 10.0 and
            0.01 < w_crit < 5.0 and 
            0.1 < f_max < 0.99 and
            -1.0 < delta_M < 1.0 and      # Prior for delta_M
            -1.0 < z_local < 0.0):        # Prior for z_local (blueshift)
        return -np.inf
    return 0.0

def log_probability_plm_final(params):
    lp = log_prior_plm_final(params)
    if not np.isfinite(lp):
        return -np.inf
    
    # Extract all parameters
    H0, omega_m_h2, z_crit, w_crit, f_max, delta_M, z_local = params
    
    # The cosmological model takes 5 parameters + FIXED_K
    cosmo_params = [H0, omega_m_h2, z_crit, w_crit, f_max, FIXED_K]
    
    # log_likelihood_plm now takes delta_M and z_local
    log_like = log_likelihood_plm(cosmo_params, delta_M=delta_M, z_local=z_local) 
    
    if not np.isfinite(log_like):
        return -np.inf
    return lp + log_like
# --- End global definitions ---

# --- Global definitions for PLM with CMB Angle Prior ---
# Global CMBLikelihood instance for prior calculation
# This needs to be outside main for pickling
# Assuming cmb_likelihood_instance is already globally defined, as it is in predict_cmb_angle.py
# If not, it needs to be defined here: cmb_likelihood_instance = CMBLikelihood()

def log_prior_plm_cmb(params):
    H0, omega_m_h2, z_crit, w_crit, f_max, delta_M, z_local = params
    
    # Standard boundaries
    if not (40 < H0 < 80 and 
            0.1 < omega_m_h2 < 0.3 and 
            0.1 < z_crit < 10.0 and
            0.01 < w_crit < 5.0 and 
            0.1 < f_max < 0.99 and
            -1.0 < delta_M < 1.0 and
            -0.1 < z_local < 0.1):
        return -np.inf
    
    # --- STRONG CMB CONSTRAINT ---
    log_prior_val = 0.0
    try:
        # Create a temporary PLM model instance with cosmological parameters
        cosmo_params = [H0, omega_m_h2, z_crit, w_crit, f_max, FIXED_K]
        temp_model = PLM(*cosmo_params)
        
        # Calculate 100*theta_s using the global cmb_likelihood_instance
        z_star = 1090.0
        r_s_comoving = cmb_likelihood_instance.calculate_sound_horizon(temp_model, z_star)
        D_A = temp_model.angular_diameter_distance(z_star)
        
        if not (np.isfinite(r_s_comoving) and np.isfinite(D_A) and D_A > 0):
            return -np.inf
        
        r_s_physical = r_s_comoving / (1.0 + z_star)
        theta_s_model = 100 * r_s_physical / D_A
        
        # Planck values
        theta_planck = 1.04109
        theta_error = 0.00030
        
        # Add Gaussian log-likelihood to the prior
        log_prior_val += -0.5 * ((theta_s_model - theta_planck) / theta_error)**2
        
    except Exception:
        return -np.inf # If calculation fails, reject the point
    
    return log_prior_val

# New log_likelihood_plm that accepts 7 parameters and passes them correctly
def log_likelihood_plm_7_params(params):
    H0, omega_m_h2, z_crit, w_crit, f_max, delta_M, z_local = params
    
    # PLM class expects cosmological parameters
    cosmo_params = [H0, omega_m_h2, z_crit, w_crit, f_max, FIXED_K]
    
    # Likelihood functions expect 'delta_M' and 'z_local'
    try:
        model = PLM(*cosmo_params)
        
        logL_sn = sn_likelihood.log_likelihood(model, delta_M=delta_M, z_local=z_local)
        logL_bao = bao_likelihood.log_likelihood(model) # BAO is not affected by delta_M or z_local
        # CMB likelihood is now part of the prior, so don't add it here.
        
        return logL_sn + logL_bao
    except Exception:
        return -np.inf

# Final log_probability function
def final_log_prob_with_cmb_prior(params):
    log_prior_val = log_prior_plm_cmb(params)
    if not np.isfinite(log_prior_val):
        return -np.inf

    log_like_val = log_likelihood_plm_7_params(params)
    if not np.isfinite(log_like_val):
        return -np.inf
    
    return log_prior_val + log_like_val

def run_optimized_mcmc(model_name, log_prob_func, initial_params, n_walkers, n_steps, n_burnin, n_cores):
    """
    Optimized MCMC runner with more efficient CPU usage
    """
    n_dim = len(initial_params)
    
    # Configure HDF5 backend
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_filename = os.path.join(results_dir, f"{model_name}_optimized_checkpoint.h5")
    backend = emcee.backends.HDFBackend(checkpoint_filename)
    
    # Check for resume
    resume = backend.initialized and backend.iteration > 0
    if resume:
        iteration = backend.iteration
        log_safe(logger, logging.INFO, f"Resuming simulation for '{model_name}' from step {iteration}.")
        initial_state = backend.get_last_sample()
        n_steps_to_run = n_steps - iteration
        if n_steps_to_run <= 0:
            log_safe(logger, logging.INFO, "Simulation already completed.")
            return backend.get_chain(discard=n_burnin, flat=True, thin=15)
    else:
        log_safe(logger, logging.INFO, f"Starting new simulation for '{model_name}'.")
        backend.reset(n_walkers, n_dim)
        initial_state = np.array(initial_params) + 1e-3 * np.random.randn(n_walkers, n_dim)
        n_steps_to_run = n_steps

    # Create pool with fewer processes for more efficient usage
    effective_cores = min(n_cores, n_walkers // 2)  # Not more than half the walkers
    log_safe(logger, logging.INFO, f"Using {effective_cores} out of {n_cores} available cores")
    
    with multiprocessing.Pool(effective_cores, initializer=init_worker) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob_func, pool=pool, backend=backend)

        log_safe(logger, logging.INFO, f"Starting optimized MCMC simulation for {model_name}...")
        start_time = time.time()
        
        # Monitor CPU usage
        cpu_percent_start = psutil.cpu_percent(interval=1)
        
        try:
            log_interval = max(1, n_steps_to_run // 10)
            
            for i, result in enumerate(sampler.sample(initial_state, iterations=n_steps_to_run, progress=True)):
                if (i + 1) % log_interval == 0 or i == n_steps_to_run - 1:
                    current_log_probs = result.log_prob
                    mean_log_prob = np.mean(current_log_probs)
                    std_log_prob = np.std(current_log_probs)
                    acceptance_fraction = np.mean(sampler.acceptance_fraction)
                    
                    # Check CPU usage
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    
                    log_safe(logger, logging.INFO,
                        f"  Step {sampler.iteration}/{n_steps} | "
                        f"Accepted: {acceptance_fraction:.2%} | "
                        f"Log_prob: {mean_log_prob:.2f}±{std_log_prob:.2f} | "
                        f"CPU: {cpu_percent:.1f}% | "
                        f"Time: {(time.time() - start_time):.1f}s"
                    )
                    
        except Exception as e:
            log_safe(logger, logging.ERROR, f"Error during simulation: {e}")
            log_safe(logger, logging.ERROR, traceback.format_exc())
            return None

        end_time = time.time()
        cpu_percent_end = psutil.cpu_percent(interval=1)
        
        log_safe(logger, logging.INFO, 
            f"Simulation finished in {(end_time - start_time) / 60:.2f} min. "
            f"Average CPU: {(cpu_percent_start + cpu_percent_end) / 2:.1f}%"
        )
        
        return sampler.get_chain(discard=n_burnin, flat=True, thin=15)

def analyze_and_plot_results(flat_samples, param_names, model_name):
    """Analyzes and visualizes MCMC results."""
    if flat_samples is None or len(flat_samples) == 0:
        log_safe(logger, logging.WARNING, f"No samples for analysis for model '{model_name}'.")
        return

    log_safe(logger, logging.INFO, f"\nAnalysis results for {model_name}:")
    log_safe(logger, logging.INFO, f"  Number of samples: {len(flat_samples)}")
    
    # Corner plot
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, f"{model_name}_optimized_corner_plot.png")
    
    fig = corner.corner(flat_samples, labels=param_names, hist_bin_factor=2, 
                       quantiles=[0.16, 0.5, 0.84], show_titles=True)
    plt.suptitle(f"Optimized Corner Plot for {model_name}", fontsize=16)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log_safe(logger, logging.INFO, f"  Corner plot: {plot_path}")
    
    # Parameter statistics
    log_safe(logger, logging.INFO, "  Results:")
    for i, name in enumerate(param_names):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        log_safe(logger, logging.INFO, f"    {name} = {mcmc[1]:.4f} +{q[1]:.4f} / -{q[0]:.4f}")


def main():
    """Main function to execute the script."""
    # Force stdout and stderr to use UTF-8 encoding on Windows
    # This is critical for displaying special characters correctly in the console
    if sys.platform == "win32" and sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    # This check is CRITICAL for multiprocessing to work correctly on Windows
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Run MCMC simulations for cosmological models.")
    parser.add_argument("--model", type=str, default="PLM", choices=["PLM", "LCDM"],
                        help="Cosmological model to test (PLM or LCDM).")
    parser.add_argument("--n_walkers", type=int, default=32,
                        help="Number of walkers in the MCMC simulation.")
    parser.add_argument("--n_steps", type=int, default=5000,
                        help="Total number of steps in the simulation.")
    parser.add_argument("--n_burnin", type=int, default=1000,
                        help="Number of steps for the 'burn-in' phase, which will be discarded.")
    parser.add_argument("--n_cores", type=int, default=-1,
                        help="Number of CPU cores (-1 means all available).")
    parser.add_argument("--plot_only", action="store_true", 
                        help="Only generate plots from existing HDF5 checkpoint, skip MCMC simulation.")
    
    args = parser.parse_args()

    # Setup environment and logging
    setup_cp1251_environment()
    global logger
    logger = setup_cp1251_logger('PLM_MCMC_Runner', logging.DEBUG) # Use DEBUG level for detailed logs

    # Setup fonts for matplotlib
    clear_font_cache()
    setup_cyrillic_fonts()

    # Optimize number of cores
    total_cores = multiprocessing.cpu_count()
    if args.n_cores == -1:
        n_cores = max(1, total_cores - 2)  # Leave 2 cores for the system
    else:
        n_cores = min(args.n_cores, total_cores)
    
    log_safe(logger, logging.INFO, "="*60)
    log_safe(logger, logging.INFO, f"OPTIMIZED MCMC RUNNER")
    log_safe(logger, logging.INFO, "="*60)
    log_safe(logger, logging.INFO, f"Model: {args.model}")
    log_safe(logger, logging.INFO, f"Walkers: {args.n_walkers}, Steps: {args.n_steps}, Burnin: {args.n_burnin}")
    log_safe(logger, logging.INFO, f"CPU Cores: {n_cores}/{total_cores}")
    log_safe(logger, logging.INFO, f"System: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")

    try:
        if args.plot_only:
            log_safe(logger, logging.INFO, f"Plot-only mode enabled. Attempting to load samples from checkpoint.")
            results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
            checkpoint_filename = os.path.join(results_dir, f"{args.model}_CMB_constrained_optimized_checkpoint.h5" if args.model == "PLM" else f"{args.model}_optimized_checkpoint.h5")
            
            if not os.path.exists(checkpoint_filename):
                log_safe(logger, logging.ERROR, f"Checkpoint file not found: {checkpoint_filename}. Cannot plot without samples.")
                return

            backend = emcee.backends.HDFBackend(checkpoint_filename, read_only=True)
            try:
                flat_samples = backend.get_chain(discard=args.n_burnin, flat=True, thin=15)
                log_safe(logger, logging.INFO, f"Successfully loaded {len(flat_samples)} samples from {checkpoint_filename}")
            except Exception as e:
                log_safe(logger, logging.ERROR, f"Error loading samples from checkpoint: {e}")
                log_safe(logger, logging.ERROR, traceback.format_exc())
                return

            if args.model == "PLM":
                param_names = ["H0", "Omega_m h^2", "z_crit", "w_crit", "f_max", "delta_M", "z_local"]
                analyze_and_plot_results(flat_samples, param_names, "PLM_CMB_constrained")
            elif args.model == "LCDM":
                param_names = ["H0", "Omega_m h^2", "Omega_b h^2", "n_s", "A_s", "tau_reio"]
                analyze_and_plot_results(flat_samples, param_names, "LCDM")
            
            log_safe(logger, logging.INFO, "\nPlot generation completed successfully!")
        else: # Normal MCMC simulation mode
            if args.model == "PLM":
                # --- NEW SCENARIO: Strong CMB Angle Prior ---
                log_safe(logger, logging.INFO, "STARTING SIMULATION WITH STRONG CMB ANGLE PRIOR")

                # The 7 free parameters are: H0, omega_m_h2, z_crit, w_crit, f_max, delta_M, z_local
                initial_params_cmb = [55.0, 0.2, 3.0, 1.0, 0.5, 0.0, -0.04] 
                param_names_cmb = ["H0", "Omega_m h^2", "z_crit", "w_crit", "f_max", "delta_M", "z_local"]
                
                # --- Starting the simulation with new settings ---
                samples = run_optimized_mcmc(
                    "PLM_CMB_constrained", # New name
                    final_log_prob_with_cmb_prior,
                    initial_params_cmb, 
                    args.n_walkers, 
                    args.n_steps, 
                    args.n_burnin, 
                    n_cores
                )
                if samples is not None:
                    analyze_and_plot_results(samples, param_names_cmb, "PLM_CMB_constrained")

            elif args.model == "LCDM":
                initial_params = [67.36, 0.14, 0.022, 0.96, 2.1e-9, 0.054]
                param_names = ["H0", "Omega_m h^2", "Omega_b h^2", "n_s", "A_s", "tau_reio"]
                
                samples = run_optimized_mcmc("LCDM", log_probability_lcdm, initial_params, 
                                           args.n_walkers, args.n_steps, args.n_burnin, n_cores)
                if samples is not None:
                    analyze_and_plot_results(samples, param_names, "LCDM")
            
            log_safe(logger, logging.INFO, "\nOptimized MCMC simulations finished successfully!")

    except Exception as e:
        logging.error("An unexpected error occurred at the top level.")
        logging.error(traceback.format_exc())
    finally:
        logging.info("======================================================")
        logging.info("            MCMC SIMULATION CONCLUDED")
        logging.info("======================================================")

# === Main executable part ===
if __name__ == "__main__":
    # This check is CRITICAL for multiprocessing to work correctly on Windows
    multiprocessing.set_start_method('spawn', force=True)
    
    main()
