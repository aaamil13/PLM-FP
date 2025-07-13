"""
Основен скрипт за MCMC симулации
=================================

Този скрипт дефинира лог-правдоподобностни функции, priors и стартира
MCMC симулации за Подобрения Линеен Модел (PLM) и стандартния ΛCDM модел
спрямо Supernovae (SN), BAO и CMB данни.

Използва `logging` за запис на прогреса и грешките и `emcee.HDFBackend`
за надеждно запазване и възстановяване на състоянието на симулацията.
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


# Деактивираме заключването на файлове за HDF5
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# Добавяме директориите на проекта в пътя за търсене на модули
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mcmc_analysis.models.plm_model_fp import PLM
from mcmc_analysis.models.lcdm_model import LCDM
from mcmc_analysis.likelihoods.sn_likelihood import SupernovaeLikelihood
from mcmc_analysis.likelihoods.bao_likelihood import BAOLikelihood
from mcmc_analysis.likelihoods.cmb_likelihood import CMBLikelihood

from utils.font_config import setup_cyrillic_fonts, clear_font_cache
from utils.encoding_utils import setup_cp1251_environment
from utils.logger_config import setup_cp1251_logger, log_safe


# Global instances - ще бъдат инициализирани в worker процесите
sn_likelihood = None
bao_likelihood = None
cmb_likelihood = None
logger = None

# === Дефиниция на лог-вероятностни функции и priors ===

def init_worker():
    """Инициализация на worker процеси"""
    global sn_likelihood, bao_likelihood, cmb_likelihood, logger
    
    # Настройваме логването за worker процеса
    logger = setup_cp1251_logger(f'Worker-{os.getpid()}', logging.INFO)
    
    try:
        # Зареждаме likelihood данните в началото на всеки worker
        sn_likelihood = SupernovaeLikelihood()
        bao_likelihood = BAOLikelihood()
        cmb_likelihood = CMBLikelihood()
        log_safe(logger, logging.INFO, f"Worker {os.getpid()} инициализиран успешно")
    except Exception as e:
        log_safe(logger, logging.ERROR, f"Грешка при инициализация на worker {os.getpid()}: {e}")
        raise

def log_prior_plm(params):
    """Лог-prior функция за PLM модела."""
    # Параметри за новия PLM-FP: H0, omega_m_h2, z_crit, w_crit, f_max, k
    H0, omega_m_h2, z_crit, w_crit, f_max, k = params
    if not (40 < H0 < 100 and 0.1 < omega_m_h2 < 0.3 and 0.1 < z_crit < 10.0 and
            0.01 < w_crit < 5.0 and 0.1 < f_max < 0.99 and 0.01 < k < 5.0):
        return -np.inf
    return 0.0

def log_likelihood_plm(params, delta_M=0.0, z_local=0.0):
    """Оптимизирана лог-likelihood функция за PLM модела."""
    global sn_likelihood, bao_likelihood, cmb_likelihood
    
    try:
        model = PLM(*params)
        
        # Изчисляваме likelihood-овете паралелно
        log_like_sn = sn_likelihood.log_likelihood(model, delta_M=delta_M, z_local=z_local) # Pass delta_M and z_local here
        log_like_bao = bao_likelihood.log_likelihood(model)
        log_like_cmb = cmb_likelihood.log_likelihood(model)
        
        total_lp = log_like_sn + log_like_bao + log_like_cmb
        
        return total_lp
        
    except Exception as e:
        return -np.inf

def log_probability_plm(params):
    """Обща лог-вероятностна функция за PLM модела."""
    lp = log_prior_plm(params)
    if not np.isfinite(lp):
        return -np.inf
        
    log_like = log_likelihood_plm(params)
    if not np.isfinite(log_like):
        return -np.inf
        
    return lp + log_like

def log_prior_lcdm(params):
    """Лог-prior функция за LCDM модела."""
    H0, omega_m_h2, omega_b_h2, n_s, A_s, tau_reio = params
    if not (50 < H0 < 100 and 0.05 < omega_m_h2 < 0.20 and 0.01 < omega_b_h2 < 0.03 and
            0.8 < n_s < 1.2 and 1.0e-9 < A_s < 5.0e-9 and 0.01 < tau_reio < 0.1):
        return -np.inf
    return 0.0

def log_likelihood_lcdm(params):
    """Оптимизирана лог-likelihood функция за LCDM модела."""
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
    """Обща лог-вероятностна функция за LCDM модела."""
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
    if not (60 < H0 < 70 and           # **СИЛНО ОГРАНИЧЕН PRIOR ЗА H0**
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

def run_optimized_mcmc(model_name, log_prob_func, initial_params, n_walkers, n_steps, n_burnin, n_cores):
    """
    Оптимизиран MCMC runner с по-ефективно използване на CPU
    """
    n_dim = len(initial_params)
    
    # Конфигуриране на HDF5 backend
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_filename = os.path.join(results_dir, f"{model_name}_optimized_checkpoint.h5")
    backend = emcee.backends.HDFBackend(checkpoint_filename)
    
    # Проверка за възобновяване
    resume = backend.initialized and backend.iteration > 0
    if resume:
        iteration = backend.iteration
        log_safe(logger, logging.INFO, f"Възстановяване на симулация за '{model_name}' от стъпка {iteration}.")
        initial_state = backend.get_last_sample()
        n_steps_to_run = n_steps - iteration
        if n_steps_to_run <= 0:
            log_safe(logger, logging.INFO, "Симулацията вече е завършена.")
            return backend.get_chain(discard=n_burnin, flat=True, thin=15)
    else:
        log_safe(logger, logging.INFO, f"Започване на нова симулация за '{model_name}'.")
        backend.reset(n_walkers, n_dim)
        initial_state = np.array(initial_params) + 1e-3 * np.random.randn(n_walkers, n_dim)
        n_steps_to_run = n_steps

    # Създаваме pool с по-малко процеси за по-ефективно използване
    effective_cores = min(n_cores, n_walkers // 2)  # Не повече от половината walker-и
    log_safe(logger, logging.INFO, f"Използваме {effective_cores} от {n_cores} налични ядра")
    
    with multiprocessing.Pool(effective_cores, initializer=init_worker) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob_func, pool=pool, backend=backend)

        log_safe(logger, logging.INFO, f"Стартиране на оптимизирана MCMC симулация за {model_name}...")
        start_time = time.time()
        
        # Мониторинг на CPU използването
        cpu_percent_start = psutil.cpu_percent(interval=1)
        
        try:
            log_interval = max(1, n_steps_to_run // 10)
            
            for i, result in enumerate(sampler.sample(initial_state, iterations=n_steps_to_run, progress=True)):
                if (i + 1) % log_interval == 0 or i == n_steps_to_run - 1:
                    current_log_probs = result.log_prob
                    mean_log_prob = np.mean(current_log_probs)
                    std_log_prob = np.std(current_log_probs)
                    acceptance_fraction = np.mean(sampler.acceptance_fraction)
                    
                    # Проверяваме CPU използването
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    
                    log_safe(logger, logging.INFO,
                        f"  Стъпка {sampler.iteration}/{n_steps} | "
                        f"Приети: {acceptance_fraction:.2%} | "
                        f"Log_prob: {mean_log_prob:.2f}±{std_log_prob:.2f} | "
                        f"CPU: {cpu_percent:.1f}% | "
                        f"Време: {(time.time() - start_time):.1f}s"
                    )
                    
        except Exception as e:
            log_safe(logger, logging.ERROR, f"Грешка по време на симулацията: {e}")
            log_safe(logger, logging.ERROR, traceback.format_exc())
            return None

        end_time = time.time()
        cpu_percent_end = psutil.cpu_percent(interval=1)
        
        log_safe(logger, logging.INFO, 
            f"Симулация завърши за {(end_time - start_time) / 60:.2f} мин. "
            f"Средно CPU: {(cpu_percent_start + cpu_percent_end) / 2:.1f}%"
        )
        
        return sampler.get_chain(discard=n_burnin, flat=True, thin=15)

def analyze_and_plot_results(flat_samples, param_names, model_name):
    """Анализира и визуализира резултатите от MCMC."""
    if flat_samples is None or len(flat_samples) == 0:
        log_safe(logger, logging.WARNING, f"Няма семпли за анализ за модел '{model_name}'.")
        return

    log_safe(logger, logging.INFO, f"\nАнализ на резултатите за {model_name}:")
    log_safe(logger, logging.INFO, f"  Брой семпли: {len(flat_samples)}")
    
    # Corner plot
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, f"{model_name}_optimized_corner_plot.png")
    
    fig = corner.corner(flat_samples, labels=param_names, hist_bin_factor=2, 
                       quantiles=[0.16, 0.5, 0.84], show_titles=True)
    plt.suptitle(f"Оптимизиран Corner Plot за {model_name}", fontsize=16)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log_safe(logger, logging.INFO, f"  Corner plot: {plot_path}")
    
    # Статистика на параметрите
    log_safe(logger, logging.INFO, "  Резултати:")
    for i, name in enumerate(param_names):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        log_safe(logger, logging.INFO, f"    {name} = {mcmc[1]:.4f} +{q[1]:.4f} / -{q[0]:.4f}")


def main():
    """Основна функция за изпълнение на скрипта."""
    # Force stdout and stderr to use UTF-8 encoding on Windows
    # This is critical for displaying special characters correctly in the console
    if sys.platform == "win32" and sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    # Тази проверка е КРИТИЧНА за правилната работа на multiprocessing под Windows
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Run MCMC simulations for cosmological models.")
    parser.add_argument("--model", type=str, default="PLM", choices=["PLM", "LCDM"],
                        help="Космологичен модел за тестване (PLM или LCDM).")
    parser.add_argument("--n_walkers", type=int, default=32,
                        help="Брой 'ходещи' в MCMC симулацията.")
    parser.add_argument("--n_steps", type=int, default=5000, # Възстановено до нормални стъпки
                        help="Общ брой стъпки в симулацията.")
    parser.add_argument("--n_burnin", type=int, default=1000, # Възстановено до нормални стъпки
                        help="Брой стъпки за 'burn-in' фазата, които ще бъдат отхвърлени.")
    parser.add_argument("--n_cores", type=int, default=-1,
                        help="Брой процесорни ядра (-1 означава всички налични).")
    
    args = parser.parse_args()

    # Настройваме средата и логването
    setup_cp1251_environment()
    global logger
    logger = setup_cp1251_logger('PLM_MCMC_Runner', logging.DEBUG) # Use DEBUG level for detailed logs

    # Настройваме фонтовете за matplotlib
    clear_font_cache()
    setup_cyrillic_fonts()

    # Оптимизираме броя ядра
    total_cores = multiprocessing.cpu_count()
    if args.n_cores == -1:
        n_cores = max(1, total_cores - 2)  # Оставяме 2 ядра за системата
    else:
        n_cores = min(args.n_cores, total_cores)
    
    log_safe(logger, logging.INFO, "="*60)
    log_safe(logger, logging.INFO, f"ОПТИМИЗИРАН MCMC RUNNER")
    log_safe(logger, logging.INFO, "="*60)
    log_safe(logger, logging.INFO, f"Модел: {args.model}")
    log_safe(logger, logging.INFO, f"Walkers: {args.n_walkers}, Steps: {args.n_steps}, Burnin: {args.n_burnin}")
    log_safe(logger, logging.INFO, f"CPU ядра: {n_cores}/{total_cores}")
    log_safe(logger, logging.INFO, f"Система: {psutil.cpu_count(logical=False)} физически, {psutil.cpu_count(logical=True)} логически")

    try:
        if args.model == "PLM":
            # --- ФИНАЛЕН СЦЕНАРИЙ: H0 свободен, k фиксирано, delta_M свободен, z_local свободен ---
            log_safe(logger, logging.INFO, "СТАРТИРАНЕ НА ФИНАЛНА СИМУЛАЦИЯ (свободен H0, фиксирано k, свободен delta_M, свободен z_local)")

            # Параметри, които ще варираме (7 на брой):
            # H0, omega_m_h2, z_crit, w_crit, f_max, delta_M, z_local
            initial_params_final = [70.0, 0.14, 2.5, 0.5, 0.85, 0.0, -0.5] # delta_M около 0, z_local около -0.5
            param_names_final = ["H0", "Omega_m h^2", "z_crit", "w_crit", "f_max", "delta_M", "z_local"]
            
            # --- Стартиране на симулацията с новите настройки ---
            samples = run_optimized_mcmc(
                "PLM_z_local", # Ново име за файловете
                log_probability_plm_final, 
                initial_params_final, 
                args.n_walkers, 
                args.n_steps, 
                args.n_burnin, 
                n_cores
            )
            if samples is not None:
                analyze_and_plot_results(samples, param_names_final, "PLM_z_local")

        elif args.model == "LCDM":
            initial_params = [67.36, 0.14, 0.022, 0.96, 2.1e-9, 0.054]
            param_names = ["H0", "Omega_m h^2", "Omega_b h^2", "n_s", "A_s", "tau_reio"]
            
            samples = run_optimized_mcmc("LCDM", log_probability_lcdm, initial_params, 
                                       args.n_walkers, args.n_steps, args.n_burnin, n_cores)
            if samples is not None:
                analyze_and_plot_results(samples, param_names, "LCDM")
        
        log_safe(logger, logging.INFO, "\nОптимизираните MCMC симулации завършиха успешно!")

    except Exception as e:
        logging.error("Възникна неочаквана грешка на най-високо ниво.")
        logging.error(traceback.format_exc())
    finally:
        logging.info("======================================================")
        logging.info("            MCMC СИМУЛАЦИЯТА ПРИКЛЮЧИ")
        logging.info("======================================================")

# === Основна изпълнима част ===
if __name__ == "__main__":
    # Тази проверка е КРИТИЧНА за правилната работа на multiprocessing под Windows
    main()
