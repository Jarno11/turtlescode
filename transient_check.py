import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from pathlib import Path

run = str(Path(__file__).resolve().parents[1].name)

plt.rcParams.update({
    "font.family": "cmr10",
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "font.size": 14,
})

lw = 2.0
u  = 0.2366431913
D  = 1

charcoal = "#333F48"
turq     = "#0892A5"
pink     = "#BE5692"


def interpolate_group(group):
    """Interpolate u and v signals to the coarsest timestep resolution within a quarter."""
    dt    = group["max_dt"][0]
    t_min = group["t"].min()
    t_max = group["t"].max()
    t_new = np.arange(t_min, t_max + dt / 2, dt)

    res = {
        "t": t_new,
        "x": group["x"][0],
        "y": group["y"][0],
        "z": group["z"][0],
        "q": group["q"][0],
    }
    for col in ["u", "v"]:
        res[col] = np.interp(t_new, group["t"], group[col])

    return pl.DataFrame(res)


def compute_chi(means, stds, mean_is_zero, tol=1e-2):
    """
    Compute the convergence statistic chi.

    Compares the mean and standard deviation across time quarters.
    Returns the maximum relative deviation (in percent) across all quarter pairs.
    """
    means_subset = np.array(means[1:])
    stds_subset  = np.array(stds[1:])
    mean_ref     = means[-1]
    std_ref      = stds[-1]

    if mean_is_zero:
        means_subset = np.zeros(3)
        mean_ref     = 1.0

    mean_diff = np.abs(np.subtract.outer(means_subset, means_subset)) / abs(mean_ref + 1e-9)
    std_diff  = np.abs(np.subtract.outer(stds_subset, stds_subset))  / (std_ref + 1e-9)

    chi = max(np.max(mean_diff), np.max(std_diff)) * 100
    return chi


def main():
    input_file = Path("solution/transient_data.csv")
    df = pl.read_csv(input_file, columns=["t", "x", "y", "z", "u", "v"])
    T  = df.select(pl.col("t").max()).item()

    times = np.linspace(50, T, 50)
    temporal_results = []

    for T_limit in times:
        df_slice = df.filter(pl.col("t") <= T_limit)

        # Divide the signal into four equal time quarters
        df_slice = df_slice.with_columns(
            q=( pl.col("t") / (T_limit / 4 + 1e-9) ).floor().clip(0, 3)
        )

        stats_slice = df_slice.group_by("q").agg(
            max_dt       = pl.col("t").unique().sort().diff().max(),
            min_t        = pl.col("t").min(),
            max_t        = pl.col("t").max(),
            count        = pl.len(),
            unique_times = pl.col("t").n_unique()
        ).sort("q")

        df_slice = df_slice.join(stats_slice, on="q")

        # Interpolate to the coarsest temporal resolution within each quarter
        df_slice_interpolated = (
            df_slice.group_by(["x", "y", "z", "q"])
            .map_groups(interpolate_group)
            .sort(["q", "t", "x", "y", "z"])
        )

        # Compute mean and std over each quarter, collapsing span and time
        stats_df_slice = (
            df_slice_interpolated
            .with_columns(x_rounded=pl.col("x").round(4))
            .group_by(["q", "x_rounded"])
            .agg(
                u_mean=pl.col("u").mean(),
                u_std =pl.col("u").std(),
                v_mean=pl.col("v").mean(),
                v_std =pl.col("v").std(),
            )
            .sort(["q", "x_rounded"])
        )

        unique_xs = stats_df_slice["x_rounded"].unique().sort()

        for x in unique_xs:
            subset = stats_df_slice.filter(pl.col("x_rounded") == x).sort("q")
            chi_u  = compute_chi(subset["u_mean"].to_numpy(), subset["u_std"].to_numpy(), mean_is_zero=False)
            chi_v  = compute_chi(subset["v_mean"].to_numpy(), subset["v_std"].to_numpy(), mean_is_zero=True)
            temporal_results.append({"time": T_limit, "x": x, "chi_u": chi_u, "chi_v": chi_v})

    df_evol = pl.DataFrame(temporal_results)

    # Plot temporal evolution of the convergence statistic
    colors = [charcoal, turq, pink]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for j, x in enumerate(unique_xs):
        color = colors[j]
        data  = df_evol.filter(pl.col("x") == x)
        ax1.plot(data["time"] * u / D, data["chi_u"], label=f'x={x:.2f}', lw=lw, color=color)
        ax2.plot(data["time"] * u / D, data["chi_v"], label=f'x={x:.2f}', lw=lw, color=color)

    for ax, label in zip([ax1, ax2], [r"$\chi(u)$ [%]", r"$\chi(v)$ [%]"]):
        ax.set_yscale('log')
        ax.set_xlabel(r"$t$ [-]")
        ax.set_ylabel(label)
        ax.set_xlim(50 * u / D, T * u / D)
        ax.set_ylim(3e0, 3e2)
        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.legend()

    plt.tight_layout()


if __name__ == "__main__":
    main()
    output_path = f"<transient_plots_dir>/{run}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
