import pyvista as pv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

def plot_simulation_snapshot(time_index, simulation_index, ds, mesh):
    camera_position = [
        (0.03, -0.07, 0.05),  # camera position (x, y, z)
        (0.025, 0.0, 0.0),  # focal point — where the camera looks
        (0.0, 0.0, 1.0),  # view-up vector
    ]

    t = ds.time[time_index].values
    snapshot = ds.isel(batch=simulation_index, time=time_index)

    plotter = pv.Plotter(shape=(1, 2), window_size=(1500, 450), notebook=True)

    plotter.subplot(0, 0)
    scalars = snapshot.heat_source.to_numpy()
    plotter.add_mesh(
        mesh,
        scalars=scalars,
        cmap="seismic",
        scalar_bar_args={"title": "Heat source [W/m^3]"},
    )
    plotter.add_text(f"Heat source field\nt={t}", font_size=10)
    plotter.camera_position = camera_position

    plotter.subplot(0, 1)
    scalars = snapshot.temperature.to_numpy()
    plotter.add_mesh(
        mesh,
        scalars=scalars,
        cmap="seismic",
        scalar_bar_args={"title": "Temperature [K]"},
    )
    plotter.add_text(f"Temperature field\nt={t}", font_size=10)

    plotter.link_views()
    plotter.show()


def plot_singular_values(snapshots, title=""):
    s = np.linalg.svd(snapshots, compute_uv=False)

    mode_idx = np.arange(1, len(s) + 1)
    mode_energy = s**2
    cum_energy_pct = 100 * np.cumsum(mode_energy) / np.sum(mode_energy)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    # Left axis: singular values
    ax1.semilogy(mode_idx, s, color="tab:blue", lw=2)
    ax1.set_xlabel("Mode index")
    ax1.set_ylabel("Singular value $s_i$", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    # ax1.set_ylim(max(1e-12, s.min()), None)
    ax1.grid(True)
    ax1.set(xscale="log")

    # Right axis: cumulative energy in modes
    ax2 = ax1.twinx()
    ax2.plot(mode_idx, cum_energy_pct, color="tab:red", lw=2)
    ax2.set_ylabel("Cumulative energy [%]", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title(f"Singular values and modal energy content for {title}")
    plt.tight_layout()
    plt.show()

def plot_modes(modes, mesh):
    camera_position = [
        (0.03, -0.07, 0.05),  # camera position (x, y, z)
        (0.025, 0.0, 0.0),  # focal point — where the camera looks
        (0.0, 0.0, 1.0),  # view-up vector
    ]

    num_modes = modes.shape[1]
    num_plots = min(16, num_modes)

    n_row = int(np.ceil(num_plots / 4))
    n_col = min(num_plots, 4)
    plotter = pv.Plotter(shape=(n_row, n_col), window_size=(500 * n_col, 300 * n_row), notebook=True)

    for i in range(num_plots):
        plotter.subplot(i // n_col, i % n_col)
        plotter.add_mesh(mesh, scalars=modes[:, i], cmap="seismic")
        plotter.add_text(f"Mode {i}", font_size=10)
        plotter.camera_position = camera_position

    plotter.link_views()
    plotter.show()

def plot_latents(latent_heat_source, latent_temperature, batch: int):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    cmap = plt.get_cmap("viridis")

    for ax, latents, title in zip(
        axes, [latent_heat_source, latent_temperature], ["Heat source", "Temperature"]
    ):
        num_modes = latents.shape[-1]
        colors = cmap(np.logspace(0, -1, num_modes))
        for i, color in enumerate(colors):
            ax.plot(latents[batch, :, i], color=color, label=i)
        ax.set(xlabel="time [s]", ylabel="Latent variables", title=f"{title} latents")
        legend_elements = [
            Line2D([0], [0], color=cmap(1.0), label="Latent 0"),
            Line2D([0], [0], color=cmap(0.5), label="...", linestyle="none"),  # spacer
            Line2D([0], [0], color=cmap(0.0), label=f"Latent {num_modes}"),
        ]
        ax.legend(handles=legend_elements)

    fig.suptitle(f"Simulation {batch}")
    plt.tight_layout()
    plt.show()