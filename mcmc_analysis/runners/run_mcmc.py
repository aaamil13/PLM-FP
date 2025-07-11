"""
Основен скрипт за MCMC симулации
=================================

Този скрипт дефинира лог-правдоподобностни функции, priors и стартира
MCMC симулации за Подобрения Линеен Модел (PLM) и стандартния ΛCDM модел
спрямо Supernovae (SN), BAO и CMB данни.
"""

import numpy as np
import emcee
import corner
import sys
import os
import time
import matplotlib.pyplot as plt
import multiprocessing # Добавяме multiprocessing

# Добавяме директориите на проекта в пътя за търсене на модули
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mcmc_analysis.models.plm_model import PLM
from mcmc_analysis.models.lcdm_model import LCDM
from mcmc_analysis.likelihoods.sn_likelihood import SupernovaeLikelihood
from mcmc_analysis.likelihoods.bao_likelihood import BAOLikelihood
from mcmc_analysis.likelihoods.cmb_likelihood import CMBLikelihood

# Инициализация на Likelihood обекти (данните се зареждат тук)
try:
    sn_likelihood = SupernovaeLikelihood()
    bao_likelihood = BAOLikelihood()
    cmb_likelihood = CMBLikelihood()
except Exception as e:
    print(f"Грешка при зареждане на Likelihood данни: {e}")
    sys.exit(1)

# === Дефиниция на лог-вероятностни функции и priors ===

def log_prior_plm(params):
    """
    Лог-prior функция за PLM модела.
    Параметри: H0, omega_m_h2, z_crit, alpha, epsilon, beta
    """
    H0, omega_m_h2, z_crit, alpha, epsilon, beta = params
    
    # Дефинираме плоски priors (uniform priors)
    if not (50 < H0 < 100 and
            0.05 < omega_m_h2 < 0.20 and
            500 < z_crit < 2000 and # z_crit около рекомбинацията
            0.1 < alpha < 5.0 and
            -0.1 < epsilon < 0.1 and # ε може да е положителен или отрицателен
            0.0 < beta < 5.0):
        return -np.inf
    
    return 0.0 # log(1) за плосък prior

def log_likelihood_plm(params):
    """
    Лог-likelihood функция за PLM модела, комбинираща всички данни.
    """
    H0, omega_m_h2, z_crit, alpha, epsilon, beta = params
    
    # Създаваме инстанция на модела с текущите параметри
    try:
        model = PLM(H0=H0, omega_m_h2=omega_m_h2, z_crit=z_crit, alpha=alpha, epsilon=epsilon, beta=beta)
    except Exception:
        return -np.inf # Връщаме -inf ако моделът не може да се инициализира

    # Изчисляваме лог-likelihood за всеки набор от данни
    lp = 0.0
    lp += sn_likelihood.log_likelihood(model)
    lp += bao_likelihood.log_likelihood(model)
    lp += cmb_likelihood.log_likelihood(model) # CMB likelihood

    return lp

def log_probability_plm(params):
    """
    Лог-вероятностна функция за PLM модела.
    """
    lp = log_prior_plm(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_plm(params)

def log_prior_lcdm(params):
    """
    Лог-prior функция за LCDM модела.
    Параметри: H0, omega_m_h2, omega_b_h2, n_s, A_s, tau_reio
    """
    H0, omega_m_h2, omega_b_h2, n_s, A_s, tau_reio = params
    
    # Дефинираме плоски priors (uniform priors)
    if not (50 < H0 < 100 and
            0.05 < omega_m_h2 < 0.20 and
            0.01 < omega_b_h2 < 0.03 and
            0.8 < n_s < 1.2 and
            1.0e-9 < A_s < 5.0e-9 and
            0.01 < tau_reio < 0.1):
        return -np.inf
    
    return 0.0 # log(1) за плосък prior

def log_likelihood_lcdm(params):
    """
    Лог-likelihood функция за LCDM модела, комбинираща всички данни.
    """
    H0, omega_m_h2, omega_b_h2, n_s, A_s, tau_reio = params
    
    # Създаваме инстанция на модела с текущите параметри
    try:
        model = LCDM(H0=H0, omega_m_h2=omega_m_h2, omega_b_h2=omega_b_h2, n_s=n_s, A_s=A_s, tau_reio=tau_reio)
    except Exception:
        return -np.inf

    # Изчисляваме лог-likelihood за всеки набор от данни
    lp = 0.0
    lp += sn_likelihood.log_likelihood(model)
    lp += bao_likelihood.log_likelihood(model)
    lp += cmb_likelihood.log_likelihood(model) # CMB likelihood

    return lp

def log_probability_lcdm(params):
    """
    Лог-вероятностна функция за LCDM модела.
    """
    lp = log_prior_lcdm(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_lcdm(params)

# === Настройки на MCMC семплера ===

def run_mcmc_for_model(model_name, log_prob_func, initial_params, n_walkers, n_steps, n_burnin, pool=None):
    """
    Стартира MCMC симулация за даден модел.
    """
    n_dim = len(initial_params)
    
    # Генериране на начални позиции за "ходещите" (walkers)
    pos = initial_params + 1e-4 * np.random.randn(n_walkers, n_dim)
    
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob_func, pool=pool) # Добавяме pool
    
    print(f"\nСтартиране на MCMC симулация за {model_name} модел...")
    start_time = time.time()
    
    # Burn-in фаза
    print(f"  Изпълнение на {n_burnin} стъпки за 'burn-in'...")
    pos, prob, state = sampler.run_mcmc(pos, n_burnin, progress=True)
    sampler.reset()
    
    # Основна фаза
    print(f"  Изпълнение на {n_steps} основни стъпки...")
    sampler.run_mcmc(pos, n_steps, progress=True)
    
    end_time = time.time()
    print(f"MCMC симулация за {model_name} завърши за {end_time - start_time:.2f} секунди.")
    
    return sampler

def analyze_and_plot_results(sampler, param_names, model_name):
    """
    Анализира и визуализира резултатите от MCMC.
    """
    flat_samples = sampler.get_chain(flat=True)
    
    print(f"\nАнализ на резултатите за {model_name} модел:")
    print(f"  Брой семпли: {len(flat_samples)}")
    
    # Построяване на corner plot
    fig = corner.corner(flat_samples, labels=param_names, hist_bin_factor=2)
    plt.suptitle(f"Corner Plot за {model_name} Модел", fontsize=16)
    plt.savefig(f"mcmc_analysis/results/{model_name}_corner_plot.png")
    plt.close(fig)
    print(f"  Corner plot запазен като mcmc_analysis/results/{model_name}_corner_plot.png")
    
    # Изчисляване на медианни стойности и 1σ грешки
    print("  Медианни стойности и 1σ грешки:")
    for i, name in enumerate(param_names):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = f"    {name} = {mcmc[1]:.3f} +{q[1]:.3f} / -{q[0]:.3f}"
        print(txt)

# === Основна изпълнима част ===
if __name__ == "__main__":
    # Параметри на симулацията
    n_walkers = 32 # Брой "ходещи"
    n_steps = 1000 # Брой стъпки след burn-in
    n_burnin = 500 # Брой стъпки за "burn-in"
    n_cores = multiprocessing.cpu_count() - 1 # Използваме всички ядра без едно

    print(f"Използваме {n_cores} ядра за паралелизация.")

    with multiprocessing.Pool(n_cores) as pool:
        # --- PLM Модел ---
        # H0, omega_m_h2, z_crit, alpha, epsilon, beta
        plm_initial_params = [70.0, 0.14, 900.0, 2.0, 0.0, 1.0] # Начални стойности
        plm_param_names = ["H0", "$\\Omega_m h^2$", "$z_{crit}$", "$\\alpha$", "$\\epsilon$", "$\\beta$"]
        
        plm_sampler = run_mcmc_for_model("PLM", log_probability_plm, plm_initial_params, n_walkers, n_steps, n_burnin, pool=pool)
        analyze_and_plot_results(plm_sampler, plm_param_names, "PLM")

        # --- LCDM Модел ---
        # H0, omega_m_h2, omega_b_h2, n_s, A_s, tau_reio
        lcdm_initial_params = [67.36, 0.1198, 0.02237, 0.9649, 2.1e-9, 0.0544] # Planck 2018 best-fit
        lcdm_param_names = ["H0", "$\\Omega_m h^2$", "$\\Omega_b h^2$", "$n_s$", "$A_s$", "$\\tau_{reio}$"]
        
        lcdm_sampler = run_mcmc_for_model("LCDM", log_probability_lcdm, lcdm_initial_params, n_walkers, n_steps, n_burnin, pool=pool)
        analyze_and_plot_results(lcdm_sampler, lcdm_param_names, "LCDM")

    print("\nВсички MCMC симулации завършиха. Резултатите са в папката 'mcmc_analysis/results'.")
