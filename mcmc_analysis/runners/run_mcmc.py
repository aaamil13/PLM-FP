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
import multiprocessing
import h5py # Добавяме h5py за checkpointing
import argparse # Добавяме argparse за CLI

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
    except Exception as e: # По-добро логиране на изключенията
        # print(f"Грешка при инициализация на PLM модел: {e}")
        return -np.inf 

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
    except Exception as e: # По-добро логиране на изключенията
        # print(f"Грешка при инициализация на LCDM модел: {e}")
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

def run_mcmc_for_model(model_name, log_prob_func, initial_params, n_walkers, n_steps, n_burnin, pool=None, checkpoint_filename=None, checkpoint_interval=100):
    """
    Стартира MCMC симулация за даден модел с checkpointing.
    """
    n_dim = len(initial_params)
    
    # Проверка за checkpoint файл
    initial_step = 0
    if checkpoint_filename and os.path.exists(checkpoint_filename):
        with h5py.File(checkpoint_filename, "r") as f:
            pos = f["pos"][()]
            initial_step = f["initial_step"][()]
            print(f"Възстановяване от checkpoint {checkpoint_filename} на стъпка {initial_step}.")
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob_func, pool=pool)
        # Продължаваме MCMC от запазената позиция
        for i, result in enumerate(sampler.sample(pos, iterations=n_steps - initial_step)): # Премахваме progress=True
            if (initial_step + i + 1) % checkpoint_interval == 0:
                print(f"  Продължаване... Стъпка: {initial_step + i + 1}/{n_steps}") # Добавяме ръчно логиране
                if checkpoint_filename:
                    # Атомарен запис
                    tmp_filename = checkpoint_filename + ".tmp"
                    with h5py.File(tmp_filename, "w") as f_tmp:
                        f_tmp.create_dataset("pos", data=result.coords)
                        f_tmp.create_dataset("initial_step", data=initial_step + i + 1)
                    os.replace(tmp_filename, checkpoint_filename)
                    print(f"  Checkpoint запазен на стъпка {initial_step + i + 1}.")
        sampler.reset() # Изчистваме преди да върнем, за да не се дублират семплите
        return sampler # Връщаме семплера след продължаване
    else:
        # Генериране на начални позиции за "ходещите" (walkers)
        # Генериране на начални позиции за "ходещите" (walkers)
        # Добавяме малко абсолютна случайна вариация
        pos = initial_params + 1e-1 * np.random.randn(n_walkers, n_dim) # Добавяме абсолютна вариация
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob_func, pool=pool)
        
        print(f"\nСтартиране на MCMC симулация за {model_name} модел...")
        start_time = time.time()
        
        # Burn-in фаза
        print(f"  Изпълнение на {n_burnin} стъпки за 'burn-in'...")
        pos, prob, state = sampler.run_mcmc(pos, n_burnin) # Премахваме progress=True
        sampler.reset()
        initial_step = n_burnin
    
    # Основна фаза
    print(f"  Изпълнение на {n_steps} основни стъпки...")
    for i, result in enumerate(sampler.sample(pos, iterations=n_steps)): # Премахваме progress=True
        if (initial_step + i + 1) % checkpoint_interval == 0:
            print(f"  Прогрес... Стъпка: {initial_step + i + 1}/{n_burnin + n_steps}") # Добавяме ръчно логиране
            if checkpoint_filename:
                # Атомарен запис
                tmp_filename = checkpoint_filename + ".tmp"
                with h5py.File(tmp_filename, "w") as f_tmp:
                    f_tmp.create_dataset("pos", data=result.coords)
                    f_tmp.create_dataset("initial_step", data=initial_step + i + 1)
                os.replace(tmp_filename, checkpoint_filename)
                print(f"  Checkpoint запазен на стъпка {initial_step + i + 1}.")
    
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
    parser = argparse.ArgumentParser(description="Run MCMC simulations for cosmological models.")
    parser.add_argument("--model", type=str, default="PLM", choices=["PLM", "LCDM"],
                        help="Specify the cosmological model to run (PLM or LCDM).")
    parser.add_argument("--n_walkers", type=int, default=32,
                        help="Number of MCMC walkers.")
    parser.add_argument("--n_steps", type=int, default=1000,
                        help="Number of MCMC steps after burn-in.")
    parser.add_argument("--n_burnin", type=int, default=500,
                        help="Number of MCMC steps for burn-in phase.")
    parser.add_argument("--n_cores", type=int, default=multiprocessing.cpu_count() - 1,
                        help="Number of CPU cores for parallelization (-1 for all but one).")
    parser.add_argument("--checkpoint_interval", type=int, default=100,
                        help="Interval (in steps) at which to save checkpoints.")
    
    args = parser.parse_args()

    n_walkers = args.n_walkers
    n_steps = args.n_steps
    n_burnin = args.n_burnin
    n_cores = args.n_cores
    checkpoint_interval = args.checkpoint_interval

    # Конфигурация за логиране на изхода
    log_file_path = os.path.join(os.path.dirname(__file__), "../results", f"mcmc_log_{int(time.time())}.txt")
    
    # Пренасочване на stdout и stderr към файл
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        with open(log_file_path, 'w') as f:
            sys.stdout = f
            sys.stderr = f
            
            print(f"Използваме {n_cores} ядра за паралелизация. Изходът се записва във файл: {log_file_path}")
            print(f"Настройки: walkers={n_walkers}, steps={n_steps}, burnin={n_burnin}, checkpoint_interval={checkpoint_interval}")

            with multiprocessing.Pool(n_cores) as pool:
                if args.model == "PLM":
                    # --- PLM Модел ---
                    # H0, omega_m_h2, z_crit, alpha, epsilon, beta
                    plm_initial_params = [70.0, 0.14, 900.0, 2.0, 0.0, 1.0] # Начални стойности
                    plm_param_names = ["H0", "$\\Omega_m h^2$", "$z_{crit}$", "$\\alpha$", "$\\epsilon$", "$\\beta$"]
                    plm_checkpoint_file = os.path.join(os.path.dirname(__file__), "../results", "plm_checkpoint.h5")
                    
                    plm_sampler = run_mcmc_for_model("PLM", log_probability_plm, plm_initial_params, n_walkers, n_steps, n_burnin, pool=pool, checkpoint_filename=plm_checkpoint_file, checkpoint_interval=checkpoint_interval)
                    analyze_and_plot_results(plm_sampler, plm_param_names, "PLM")
                elif args.model == "LCDM":
                    # --- LCDM Модел ---
                    # H0, omega_m_h2, omega_b_h2, n_s, A_s, tau_reio
                    lcdm_initial_params = [67.36, 0.1198, 0.02237, 0.9649, 2.1e-9, 0.0544] # Planck 2018 best-fit
                    lcdm_param_names = ["H0", "$\\Omega_m h^2$", "$\\Omega_b h^2$", "$n_s$", "$A_s$", "$\\tau_{reio}$"]
                    lcdm_checkpoint_file = os.path.join(os.path.dirname(__file__), "../results", "lcdm_checkpoint.h5")
                    
                    lcdm_sampler = run_mcmc_for_model("LCDM", log_probability_lcdm, lcdm_initial_params, n_walkers, n_steps, n_burnin, pool=pool, checkpoint_filename=lcdm_checkpoint_file, checkpoint_interval=checkpoint_interval)
                    analyze_and_plot_results(lcdm_sampler, lcdm_param_names, "LCDM")
                
            print("\nВсички MCMC симулации завършиха. Резултатите са в папката 'mcmc_analysis/results'.")

    finally:
        # Възстановяване на оригиналния stdout и stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"\nMCMC симулацията приключи. Пълният изход е записан във файл: {log_file_path}")
