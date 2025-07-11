"""
Enhanced Linear Cosmological Model with Dynamic Time Dilation
==============================================================

This module implements the theoretical framework for linear cosmological model
with dynamic time dilation factor dτ/dt that varies with energy density.

Mathematical Framework:
- Geometric expansion: a(t) = k * t
- Physical time dilation: dτ/dt = [1 + (ρ/ρ_crit)^α]^(-1)
- Observable distances through numerical integration

Author: Linear Cosmology Research Project
"""

import numpy as np
import scipy.integrate as integrate
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

class EnhancedLinearUniverse:
    """
    Enhanced Linear Universe model with dynamic time dilation
    """
    
    def __init__(self, H0=70.0, Omega_m=0.3, Omega_r=1e-5, alpha=2.0):
        """
        Initialize enhanced linear universe model
        
        Parameters:
        -----------
        H0 : float
            Hubble constant today [km/s/Mpc]
        Omega_m : float
            Matter density parameter today
        Omega_r : float
            Radiation density parameter today
        alpha : float
            Power index for time dilation suppression
        """
        self.H0 = H0  # km/s/Mpc
        self.Omega_m = Omega_m
        self.Omega_r = Omega_r
        self.alpha = alpha
        
        # Fundamental constants
        self.c = 299792.458  # km/s
        self.Mpc_to_km = 3.086e19  # km per Mpc
        
        # Time and density scaling
        self.t0 = 1.0  # Current time in units where H(t) = 1/t
        self.rho_crit_0 = 1.0  # Critical density today (normalized)
        
        # Age of universe in this model
        self.age_Gyr = 13.8  # Gyr (approximately)
        
    def density_evolution(self, z):
        """
        Calculate total energy density at redshift z
        
        Parameters:
        -----------
        z : float or array
            Redshift
            
        Returns:
        --------
        rho : float or array
            Total energy density normalized to critical density today
        """
        # Matter density: ρ_m ∝ (1+z)³
        rho_m = self.Omega_m * (1 + z)**3
        
        # Radiation density: ρ_r ∝ (1+z)⁴
        rho_r = self.Omega_r * (1 + z)**4
        
        return rho_m + rho_r
    
    def time_dilation_factor(self, z):
        """
        Calculate time dilation factor dτ/dt at redshift z
        
        Parameters:
        -----------
        z : float or array
            Redshift
            
        Returns:
        --------
        dtau_dt : float or array
            Time dilation factor dτ/dt
        """
        rho = self.density_evolution(z)
        return 1.0 / (1.0 + (rho / self.rho_crit_0)**self.alpha)
    
    def cosmic_time(self, z):
        """
        Calculate cosmic time t at redshift z
        For linear model: t(z) = t₀ / (1+z)
        
        Parameters:
        -----------
        z : float or array
            Redshift
            
        Returns:
        --------
        t : float or array
            Cosmic time in units where t₀ = 1
        """
        return self.t0 / (1.0 + z)
    
    def comoving_distance_integrand(self, t, z_final):
        """
        Integrand for comoving distance calculation
        
        Parameters:
        -----------
        t : float
            Cosmic time
        z_final : float
            Final redshift for integration
            
        Returns:
        --------
        integrand : float
            c * (dτ/dt) for integration
        """
        # Convert time back to redshift
        z = self.t0 / t - 1.0
        
        # Avoid negative redshifts
        if z < 0:
            z = 0.0
            
        dtau_dt = self.time_dilation_factor(z)
        return self.c * dtau_dt
    
    def comoving_distance(self, z):
        """
        Calculate comoving distance to redshift z through numerical integration
        
        Parameters:
        -----------
        z : float
            Redshift
            
        Returns:
        --------
        d_c : float
            Comoving distance [Mpc]
        """
        if z <= 0:
            return 0.0
            
        # Integration limits
        t_z = self.cosmic_time(z)
        t_0 = self.t0
        
        # Numerical integration
        result, _ = integrate.quad(
            self.comoving_distance_integrand,
            t_z, t_0,
            args=(z,),
            limit=1000
        )
        
        # Convert to Mpc
        return result * self.t0 / self.H0
    
    def angular_diameter_distance(self, z):
        """
        Calculate angular diameter distance
        
        Parameters:
        -----------
        z : float
            Redshift
            
        Returns:
        --------
        d_A : float
            Angular diameter distance [Mpc]
        """
        d_c = self.comoving_distance(z)
        return d_c / (1.0 + z)
    
    def luminosity_distance(self, z):
        """
        Calculate luminosity distance
        
        Parameters:
        -----------
        z : float
            Redshift
            
        Returns:
        --------
        d_L : float
            Luminosity distance [Mpc]
        """
        d_A = self.angular_diameter_distance(z)
        return d_A * (1.0 + z)**2
    
    def distance_modulus(self, z):
        """
        Calculate distance modulus
        
        Parameters:
        -----------
        z : float
            Redshift
            
        Returns:
        --------
        mu : float
            Distance modulus [mag]
        """
        d_L = self.luminosity_distance(z)
        return 5.0 * np.log10(d_L) + 25.0
    
    def plot_time_dilation_evolution(self, z_max=10.0):
        """
        Plot the evolution of time dilation factor
        
        Parameters:
        -----------
        z_max : float
            Maximum redshift to plot
        """
        z_range = np.logspace(-2, np.log10(z_max), 1000)
        dtau_dt = self.time_dilation_factor(z_range)
        
        plt.figure(figsize=(10, 6))
        plt.loglog(z_range, dtau_dt, 'b-', linewidth=2, label='dτ/dt')
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='dτ/dt = 1')
        plt.axvline(x=900, color='g', linestyle='--', alpha=0.7, label='z = 900 (късна епоха)')
        plt.axvline(x=1300, color='orange', linestyle='--', alpha=0.7, label='z = 1300 (рекомбинация)')
        
        plt.xlabel('Червено отместване z')
        plt.ylabel('Времеви дилатационен фактор dτ/dt')
        plt.title(f'Еволюция на времевия дилатационен фактор (α = {self.alpha})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_density_evolution(self, z_max=10.0):
        """
        Plot the evolution of energy density
        
        Parameters:
        -----------
        z_max : float
            Maximum redshift to plot
        """
        z_range = np.logspace(-2, np.log10(z_max), 1000)
        rho_total = self.density_evolution(z_range)
        rho_matter = self.Omega_m * (1 + z_range)**3
        rho_radiation = self.Omega_r * (1 + z_range)**4
        
        plt.figure(figsize=(10, 6))
        plt.loglog(z_range, rho_total, 'k-', linewidth=2, label='Обща плътност')
        plt.loglog(z_range, rho_matter, 'b--', linewidth=2, label='Материя')
        plt.loglog(z_range, rho_radiation, 'r--', linewidth=2, label='Лъчение')
        plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='ρ_crit')
        
        plt.xlabel('Червено отместване z')
        plt.ylabel('Енергийна плътност (ρ/ρ_crit)')
        plt.title('Еволюция на енергийната плътност')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def compare_distance_measures(self, z_max=2.0):
        """
        Compare different distance measures
        
        Parameters:
        -----------
        z_max : float
            Maximum redshift to plot
        """
        z_range = np.linspace(0.01, z_max, 100)
        
        # Calculate distances
        d_c = np.array([self.comoving_distance(z) for z in z_range])
        d_A = np.array([self.angular_diameter_distance(z) for z in z_range])
        d_L = np.array([self.luminosity_distance(z) for z in z_range])
        
        plt.figure(figsize=(10, 6))
        plt.plot(z_range, d_c, 'b-', linewidth=2, label='Комовингово разстояние d_c')
        plt.plot(z_range, d_A, 'g-', linewidth=2, label='Ъглово разстояние d_A')
        plt.plot(z_range, d_L, 'r-', linewidth=2, label='Светимостно разстояние d_L')
        
        plt.xlabel('Червено отместване z')
        plt.ylabel('Разстояние [Mpc]')
        plt.title('Сравнение на различни мерки за разстояние')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


def test_enhanced_model():
    """
    Test the enhanced linear model with different α values
    """
    print("Тестване на подобрения линеен модел...")
    
    # Test different alpha values
    alpha_values = [0.5, 1.0, 2.0, 3.0]
    
    for alpha in alpha_values:
        print(f"\nТестване с α = {alpha}:")
        
        model = EnhancedLinearUniverse(alpha=alpha)
        
        # Test at specific redshifts
        test_redshifts = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        for z in test_redshifts:
            rho = model.density_evolution(z)
            dtau_dt = model.time_dilation_factor(z)
            d_L = model.luminosity_distance(z)
            
            print(f"  z = {z}: ρ/ρ_crit = {rho:.3f}, dτ/dt = {dtau_dt:.6f}, d_L = {d_L:.1f} Mpc")


if __name__ == "__main__":
    test_enhanced_model() 