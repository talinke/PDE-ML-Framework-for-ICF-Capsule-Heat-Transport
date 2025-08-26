import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
from matplotlib.ticker import LogFormatter
from matplotlib.ticker import LogFormatterSciNotation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# load data
ICFdata = np.loadtxt("ICFdata.txt", comments='#', delimiter=None, usecols=(0,1,2,3))
alpha = ICFdata[:, 0]
flux  = ICFdata[:, 1]
T0    = ICFdata[:, 2]

# log-transform inputs and output
log_alpha = np.log(alpha)
log_flux  = np.log(flux)
log_T0    = np.log(T0)

# normalize inputs to [0,1]
alpha_min, alpha_max = log_alpha.min(), log_alpha.max()
flux_min, flux_max   = log_flux.min(), log_flux.max()

alpha_norm = (log_alpha - alpha_min) / (alpha_max - alpha_min)
flux_norm  = (log_flux - flux_min) / (flux_max - flux_min)
X_norm = np.column_stack((alpha_norm, flux_norm))

# fit 2D GP
kernel = RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-3, 1e3)) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-10, 1e1))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
gp.fit(X_norm, log_T0)

print("Optimized kernel:", gp.kernel_)

# prediction grid
alpha_vals = np.logspace(np.log10(alpha.min()*0.5), np.log10(alpha.max()*2), 100)
flux_vals  = np.logspace(np.log10(flux.min()*0.5), np.log10(flux.max()*2), 100)
alpha_grid, flux_grid = np.meshgrid(alpha_vals, flux_vals)

# Normalize grid
alpha_grid_norm = (np.log(alpha_grid) - alpha_min) / (alpha_max - alpha_min)
flux_grid_norm  = (np.log(flux_grid)  - flux_min) / (flux_max - flux_min)
X_grid_norm = np.column_stack((alpha_grid_norm.ravel(), flux_grid_norm.ravel()))

# predict log(T0) and std
mean_log_pred, std_log_pred = gp.predict(X_grid_norm, return_std=True)
mean_log_grid = mean_log_pred.reshape(alpha_grid.shape)
std_log_grid  = std_log_pred.reshape(alpha_grid.shape)

# Back-transform to linear T0
mean_pred = np.exp(mean_log_grid)
std_pred  = np.exp(mean_log_grid + std_log_grid) - mean_pred
# normalize based on local mean
std_pred = std_pred / (np.abs(mean_pred))


# plot mean temperature (log scale)
plt.figure(figsize=(10,7))
cp = plt.contourf(alpha_grid, flux_grid, mean_pred, levels=100, cmap='viridis', norm=LogNorm(vmin=mean_pred.min(), vmax=mean_pred.max()))
plt.colorbar(cp, label='Temperature (K)', format=LogFormatter(base=10))
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Alpha (m^2/s)')
plt.ylabel('Flux (K m/s)')
plt.title('Mean Outer Ablator Temperature')
plt.scatter(alpha, flux, c='red', marker='.', label='Training points')
plt.legend()
plt.tight_layout()
plt.savefig(f'GP_T0_2D_mean.png')
plt.show()


# plot uncertainty (log scale)
plt.figure(figsize=(10,7))
#cp = plt.contourf(alpha_grid, flux_grid, std_pred, levels=50, cmap='plasma', norm=LogNorm(vmin=std_pred.min(), vmax=std_pred.max()))
cp = plt.contourf(alpha_grid, flux_grid, std_pred, levels=80, cmap='hsv')
plt.colorbar(cp, label='Normalized Standard Deviation')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Alpha (m^2/s)')
plt.ylabel('Flux (K m/s)')
plt.scatter(alpha, flux, c='red', marker='.', label='Training points')
plt.legend()
plt.tight_layout()
plt.savefig(f'GP_T0_2D_std.png')
plt.show()
