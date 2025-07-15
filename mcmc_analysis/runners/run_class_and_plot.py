import numpy as np
import sys
import os
import logging
import matplotlib.pyplot as plt
import subprocess
import glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

def run_class_and_read_output(param_file, output_prefix, model_label):
    """Runs CLASS and reads its lensed Cls output."""
    logging.info(f"Стартиране на CLASS за {model_label}...")
    
    class_executable = 'class'
    
    # Remove old output files before running CLASS for this model
    old_files = glob.glob(output_prefix + '*.dat')
    for f in old_files:
        os.remove(f)

    command = [class_executable, param_file]
    logging.info(f"Изпълнение на команда: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        logging.info(f"CLASS за {model_label} завърши успешно.")
        logging.debug(f"CLASS stdout ({model_label}):\n{result.stdout}")
    except FileNotFoundError:
        logging.error(f"Грешка: Изпълнимият файл '{class_executable}' не е намерен.")
        logging.error("Уверете се, че сте инсталирали CLASS и че 'class' е във вашия системен PATH.")
        return None, None
    except subprocess.CalledProcessError as e:
        logging.error(f"CLASS за {model_label} се провали с грешка (return code {e.returncode}):")
        logging.error(f"STDOUT:\n{e.stdout}")
        logging.error(f"STDERR:\n{e.stderr}")
        return None, None

    # Read the results from the output file
    list_of_files = glob.glob(output_prefix + '*_cl.dat')
    if not list_of_files:
        logging.error(f"Не са намерени изходни файлове от CLASS за {model_label} с префикс '{output_prefix}'.")
        return None, None
    
    output_file = max(list_of_files, key=os.path.getctime) # Get the latest file by creation time
    
    logging.info(f"Четене на резултати от: {output_file} за {model_label}")
    
    data = np.loadtxt(output_file)
    ll = data[:, 0]      # Колона 1: l
    dl_tt = data[:, 1]   # Колона 2: D_l^TT = l(l+1)C_l/2pi
    
    return ll, dl_tt

def main():
    logging.info("Стартиране на CLASS за PLM-Proxy и ΛCDM-Reference моделите...")
    
    # Ensure the output directory for CLASS is created
    class_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/class_output"))
    os.makedirs(class_output_dir, exist_ok=True)

    # --- PLM-Proxy ---
    plm_param_file = os.path.join(os.path.dirname(__file__), 'test_plm.ini')
    plm_output_prefix = os.path.join(class_output_dir, 'plm_test') # Removed trailing underscore to match test_plm.ini
    ll_plm, dimensionless_dl_plm = run_class_and_read_output(plm_param_file, plm_output_prefix, "PLM-Proxy")
    if ll_plm is None:
        logging.error("PLM-Proxy изчисленията не успяха. Прекъсване.")
        return

    # --- ΛCDM-Reference ---
    lcdm_ref_param_file = os.path.join(os.path.dirname(__file__), 'lcdm_reference.ini')
    lcdm_ref_output_prefix = os.path.join(class_output_dir, 'lcdm_reference_')
    ll_lcdm_ref, dimensionless_dl_lcdm_ref = run_class_and_read_output(lcdm_ref_param_file, lcdm_ref_output_prefix, "Reference LCDM")
    if ll_lcdm_ref is None:
        logging.error("Reference LCDM изчисленията не успяха. Прекъсване.")
        return

    # Convert dimensionless D_l to μK^2 for both models
    T_cmb = 2.7255  # in Kelvin
    T_cmb_microK = T_cmb * 1e6
    dl_tt_plm = dimensionless_dl_plm * (T_cmb_microK**2)
    dl_tt_lcdm_ref = dimensionless_dl_lcdm_ref * (T_cmb_microK**2)

    # 4. Зареждане на реалните данни от Planck
    planck_data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../mcmc_analysis/data/planck_binned_data.txt'))
    if not os.path.exists(planck_data_file):
        logging.error(f"Не са намерени данните от Planck в '{planck_data_file}'")
        return
        
    planck_data = np.loadtxt(planck_data_file)
    planck_ell = planck_data[:, 0]
    planck_dl = planck_data[:, 1]
    planck_err_down = planck_data[:, 2]
    planck_err_up = planck_data[:, 3]

    # 5. Визуализация
    logging.info("Генериране на сравнителна графика...")
    plt.figure(figsize=(12, 8))
    
    plt.errorbar(planck_ell, planck_dl, yerr=[planck_err_down, planck_err_up], fmt='o',
                 markersize=4, color='crimson', ecolor='lightcoral', label='Planck 2018 Data')
                 
    # Чертаем PLM-Proxy и ΛCDM-Reference моделите
    plt.plot(ll_plm, dl_tt_plm, color='royalblue', linewidth=2, label='PLM-Proxy Модел')
    plt.plot(ll_lcdm_ref, dl_tt_lcdm_ref, color='green', linestyle='--', linewidth=2, label='ΛCDM (CLASS Reference)')
    
    plt.xscale('log')
    plt.xlabel('Multipole moment $\ell$')
    plt.ylabel('$D_\\ell^{TT}$ [$\\mu K^2$]')
    plt.title('CMB Power Spectrum: PLM-Proxy vs ΛCDM Reference vs Planck Data')
    plt.legend()
    plt.grid(True)
    plt.xlim(2, 2500)
    plt.ylim(0, max(np.max(dl_tt_plm) if ll_plm.size > 0 else 0, np.max(dl_tt_lcdm_ref) if ll_lcdm_ref.size > 0 else 0, np.max(planck_dl))*1.2) # Adjust ylim dynamically
    
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    plot_path = os.path.join(results_dir, "cmb_spectrum_final_comparison.png") # New filename
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    logging.info(f"Сравнителна графика запазена в: {plot_path}")

if __name__ == "__main__":
    main()
