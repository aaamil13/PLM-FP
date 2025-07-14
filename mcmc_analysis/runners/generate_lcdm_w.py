import numpy as np
import os
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

def main():
    logging.info("Генериране на ефективен w(z) за ΛCDM модела...")

    # Для ΛCDM, w(z) = -1 и cs2(z) = 0
    z_max = 3000
    z_grid = np.geomspace(1e-4, z_max, num=500) # Логаритмична скала

    w_lcdm = np.full_like(z_grid, -1.0) # w(z) = -1 for Lambda-CDM
    cs2_lcdm = np.full_like(z_grid, 0.0) # c_s^2 = 0 for Lambda-CDM

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, "lcdm_effective_w.txt")
    
    np.savetxt(output_filename, np.c_[z_grid, w_lcdm, cs2_lcdm], header="z w cs2", fmt="%.8f")
    
    logging.info(f"Успешно генериран файл с ефективен w(z) и cs2(z) за ΛCDM: {output_filename}")

    # Опционална визуализация
    plt.figure()
    plt.plot(z_grid, w_lcdm)
    plt.xscale('log')
    plt.xlabel("Redshift (z)")
    plt.ylabel("Effective w(z)")
    plt.title("Ефективен параметър на тъмната енергия за ΛCDM модела")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "lcdm_effective_w.png"))
    plt.close()
    logging.info(f"Графика на w(z) за ΛCDM е запазена.")

if __name__ == "__main__":
    main()
