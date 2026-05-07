import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import pandas as pd


class SobolSampler:
    def __init__(self, limits, n_options, seed=42):
        self.limits = limits
        self.n_options = n_options
        self.seed = seed
        self.labels = ['Reynolds', 'alpha', 'a', 'n']
        self.n_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    def generate(self, m):
        """Generate m scrambled Sobol samples and apply parameter transformations."""
        sampler = qmc.Sobol(d=len(self.limits), seed=self.seed, scramble=True)
        samples = sampler.random(m)

        l_bounds = [v[0] for v in self.limits.values()]
        u_bounds = [v[1] for v in self.limits.values()]
        samples = qmc.scale(samples, l_bounds, u_bounds)

        re    = np.round(samples[:, 0])
        alpha = np.round(samples[:, 1])
        a_val = np.round(10 ** samples[:, 2], 3)

        n_idx   = np.clip(np.floor(samples[:, 3]).astype(int), 0, len(self.n_options) - 1)
        n_final = np.array([self.n_options[i] for i in n_idx])

        return np.column_stack([re, alpha, a_val, n_final])

    def save(self, data, filename):
        pd.DataFrame(data, columns=self.labels).to_csv(filename, index=False)
        print(f"Saved {len(data)} samples to {filename}")

    def plot(self, data):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        titles = ['Reynolds (Linear)', 'alpha (Linear)', 'a (Log-Uniform)', 'n (Discrete)']

        for i, ax in enumerate(axes):
            if i == 1:
                ax.hist(data[:, i], bins=np.arange(0, 47, 2) - 0.5,
                        color='teal', edgecolor='black', alpha=0.7)
            elif i < 3:
                ax.hist(data[:, i], bins=20, color='teal', edgecolor='black', alpha=0.7)
            else:
                vals, counts = np.unique([str(v) for v in data[:, i]], return_counts=True)
                ax.bar(vals, counts, color=self.n_colors, edgecolor='black')
            ax.set_title(titles[i])

        plt.tight_layout()
        plt.show()

    def plot_3d(self, data):
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        x_min, x_max = 1000, 3900
        y_min, y_max = 0, 45
        z_min, z_max = np.min(data[:, 2]), np.max(data[:, 2])

        s_m    = 50
        s_proj = 20

        for i, n_val in enumerate(self.n_options):
            mask = (np.isinf(data[:, 3].astype(float))
                    if np.isinf(n_val)
                    else data[:, 3].astype(float) == n_val)
            subset = data[mask]

            ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2],
                       color=self.n_colors[i], label=f"n={n_val}",
                       s=s_m, edgecolors='w', linewidth=0.3, alpha=1)

            # Wall projections
            ax.scatter(np.full_like(subset[:, 0], x_min), subset[:, 1], subset[:, 2],
                       color='grey', s=s_proj, alpha=0.15)
            ax.scatter(subset[:, 0], np.full_like(subset[:, 1], y_max), subset[:, 2],
                       color='grey', s=s_proj, alpha=0.15)
            ax.scatter(subset[:, 0], subset[:, 1], np.full_like(subset[:, 2], z_min),
                       color='grey', s=s_proj, alpha=0.15)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        ax.set_xlabel('Reynolds')
        ax.set_ylabel('alpha')
        ax.set_zlabel('a')
        ax.set_title("Sobol Samples with Wall Projections")
        ax.view_init(elev=20, azim=-35)
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

        plt.tight_layout()
        plt.show()
