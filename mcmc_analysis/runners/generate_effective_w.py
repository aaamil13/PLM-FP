# Файл: mcmc_analysis/runners/generate_effective_w.py

import numpy as np
import sys
import os
import logging
import emcee
import matplotlib.pyplot as plt # Added for plotting

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from mcmc_analysis.models.plm_model_fp import PLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

def get_best_fit_from_mcmc(hdf5_file, n_burnin=1000):
    """Извлича най-добрите параметри (медианата) от HDF5 файла."""
    try:
        backend = emcee.backends.HDFBackend(hdf5_file, read_only=True)
        flat_samples = backend.get_chain(discard=n_burnin, flat=True)
        best_params = np.percentile(flat_samples, 50, axis=0)
        return best_params
    except Exception as e:
        logging.error(f"Не може да се зареди HDF5 файл '{hdf5_file}': {e}")
        return None

def main():
    logging.info("Генериране на ефективен w(z) от най-добрия PLM модел...")

    # 1. Зареждане на параметрите
    plm_hdf5_file = os.path.join(os.path.dirname(__file__), '../results/PLM_CMB_constrained_optimized_checkpoint.h5')
    params = get_best_fit_from_mcmc(plm_hdf5_file, n_burnin=1000)
    if params is None: return

    # Extract cosmological parameters (first 5) and fixed k
    # From run_mcmc.py, the order is: H0, omega_m_h2, z_crit, w_crit, f_max, delta_M, z_local
    cosmo_params_plm = params[:-2] # H0, omega_m_h2, z_crit, w_crit, f_max
    FIXED_K = 0.01 # Consistent with run_mcmc.py
    
    # The PLM model expects: H0, omega_m_h2, z_crit, w_crit, f_max, k
    model = PLM(*cosmo_params_plm, k=FIXED_K) # Pass k as a keyword argument

    H0_model = model.params['H0'] # H0 from the best-fit parameters
    Omega_m_model = model.Omega_m
    Omega_r_model = model.Omega_r # Omega_r is calculated by the model

    # 2. Генериране на H(z) таблица
    z_max = 3000
    z_grid = np.geomspace(1e-4, z_max, num=500) # Логаритмична скала
    h_plm = np.array([model.H_of_z(z) for z in z_grid])

    # 3. Изчисляване на rho_de(z)
    # H(z)² = H₀² * [ Ω_m(1+z)³ + Ω_r(1+z)⁴ + Ω_de(z) ]
    # Ω_de(z) = (H(z)_PLM / H₀_PLM)² - Ω_m(1+z)³ - Ω_r(1+z)⁴
    h_sq_plm_norm = (h_plm / H0_model)**2 # (H(z)/H0)^2
    omega_de = h_sq_plm_norm - Omega_m_model * (1 + z_grid)**3 - Omega_r_model * (1 + z_grid)**4
    
    # Защита от отрицателни стойности, които може да се получат от числени грешки
    # Отрицателни стойности за плътност са нефизични
    omega_de[omega_de < 1e-10] = 1e-10 # Set a small positive floor instead of 1e-9

    # 4. Изчисляване на производната d(rho_de)/dz числено
    # w(z) = -1 - (1/3) * (1+z) * (1/rho_de(z)) * d(rho_de(z))/dz
    # Numerically differentiate rho_de with respect to z
    # Use central difference for internal points, forward/backward for endpoints
    d_omega_de_dz = np.gradient(omega_de, z_grid)
    
    # 5. Изчисляване на w_eff(z)
    # Ensure no division by zero for omega_de
    w_eff = -1.0 - (1.0 / 3.0) * (1 + z_grid) * (d_omega_de_dz / omega_de)
    
    # Add cs2 column (sound speed squared)
    # For a perfect fluid, c_s^2 = 1.0. For more complex fluids, it can be different.
    # User's example had 1.0 for cs2.
    cs2_eff = np.ones_like(z_grid) * 1.0 # Assuming cs2 = 1.0 for simplicity

    # 6. Записване на файла
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    output_filename = os.path.join(output_dir, "plm_effective_w.txt")
    
    # Добавяме трета колона за c_s^2 = 1.0
    cs2_fld = np.ones_like(w_eff)
    
    np.savetxt(output_filename, np.c_[z_grid, w_eff, cs2_fld], header="z w_eff(z) cs2_fld(z)", fmt="%.8f")
    
    logging.info(f"Успешно генериран файл с 3 колони: {output_filename}")
    
    # Опционална визуализация
    plt.figure()
    plt.plot(z_grid, w_eff)
    plt.xscale('log')
    plt.xlabel("Redshift (z)")
    plt.ylabel("Effective w(z)")
    plt.title("Ефективен параметър на тъмната енергия за PLM модела")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "plm_effective_w.png"))
    plt.close()
    logging.info(f"Графика на w(z) е запазена.")

if __name__ == "__main__":
    main()
