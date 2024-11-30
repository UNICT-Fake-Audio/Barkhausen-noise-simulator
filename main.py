import matplotlib.pyplot as plt
import numba
import numpy as np
from tqdm.notebook import tqdm
from matplotlib.gridspec import GridSpec
import sys

from convert import convert_into_audio

np.random.seed(0)

@numba.jit(nopython=True, cache=True)
def run_krfism_simulation(J: list, f: list, b: float, temperature: float, s_init, steps_number: int, save_samples=False) -> tuple[np.ndarray, np.ndarray]:
    """
    Run kinetic random field Ising model with Glauber dynamics.

    Parameters:
        J: array_like
            Exchange coupling matrix, (Nsites, Nsites)
        f: array_like
            Random field (Nsites)
        b: float
            External magnetic field
        temperature:float
            temperature
        s_init: array_like
            Initial configuration
        steps_number: int
            Number of steps
        save_samples: bool
            Whether to save samples or not

    Returns:
        m: array_like
            Average magnetization (Nsites)
        s: array_like
            Final configuration (Nsites) or samples (steps_number, Nsites) if save_samples=True
    """

    Nsites = J.shape[0]

    s = s_init
    m = np.zeros(steps_number)

    if save_samples:
        samples = np.zeros((steps_number, Nsites))
        samples[0] = s

    for n in range(steps_number):
        idx = np.random.randint(0, Nsites)
        Delta_E = 2 * (J[idx] @ s + f[idx] + b) * s[idx]

        if temperature == 0:
            if Delta_E < 0:
                s[idx] *= -1

        elif 1 / 2 * (1 - np.tanh(Delta_E / (2 * temperature))) > np.random.rand():
            s[idx] *= -1

        m[n] = np.mean(s)

        if save_samples:
            samples[n] = s

    if not save_samples:
        samples = s.reshape(1, -1)

    return m, samples


def hystersis_cycle_athermal(J: list, f: list, b_ax: list, steps_number: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate random field Ising model on an hysteresis cycle at zero temperature.

        Parameters:
        J: array_like
            Exchange coupling matrix, (Nsites, Nsites)
        f: array_like
            Random field (Nsites)
        b_ax: array_like
            External magnetic field axis
        steps_number: int
            Number of steps

    Returns:
        m_up: array_like
            Average magnetization (Nsites)
        s_up : array_like
            Final configuration (Nsites)
        m_down: array_like
            Average magnetization (Nsites)
        s_down : array_like
            Final configuration (Nsites)
    """

    b_N = len(b_ax)
    Nsites = J.shape[0]

    s_up = np.zeros((b_N, Nsites))
    m_up = np.zeros((b_N, steps_number))

    s_down = np.zeros((b_N, Nsites))
    m_down = np.zeros((b_N, steps_number))

    # Up ramp
    s_init = -np.ones(Nsites)
    for b_idx in range(b_N):
        b = b_ax[b_idx]
        m_up[b_idx], s_up[b_idx] = run_krfism_simulation(
            J, f, b, temperature=0, s_init=s_init, steps_number=steps_number, save_samples=False
        )
        s_init = s_up[b_idx]

    # Down ramp
    s_init = s_up[-1]
    for b_idx in range(b_N - 1, -1, -1):
        b = b_ax[b_idx]
        m_down[b_idx], s_down[b_idx] = run_krfism_simulation(
            J, f, b, temperature=0, s_init=s_init, steps_number=steps_number, save_samples=False
        )
        s_init = s_down[b_idx]

    return m_up, s_up, m_down, s_down


def disorder_averged_hysteresis_cycle_athermal(J: list, f_gf: list, f_sn: int, b_ax: list, steps_number: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate random field Ising model on an hysteresis cycle at zero temperature for many disorder realizations.

        Parameters:
        J: array_like
            Exchange coupling matrix, (Nsites, Nsites)
        f_gf: array_like
            Random field generating function
        f_sn: int
            Number of realizations
        b_ax: array_like
            External magnetic field axis
        steps_number: int
            Number of steps

    Returns:
        m_up_s: array_like
            Average magnetization (Nsites)
        m_down_s: array_like
            Average magnetization (Nsites)
    """

    b_N = len(b_ax)
    m_up_s = np.zeros((f_sn, b_N))
    m_down_s = np.zeros((f_sn, b_N))

    Nsites = J.shape[0]

    for i in tqdm(range(f_sn)):
        f = f_gf(Nsites)

        # m_up, s_up, m_down, s_down
        m_up, _, m_down, _ = hystersis_cycle_athermal(J, f, b_ax, steps_number)

        m_up_s[i] = m_up[:, -1]
        m_down_s[i] = m_down[:, -1]

    return m_up_s, m_down_s

args = sys.argv

grid_len = 10
if len(args) > 1:
    grid_len = int(sys.argv[1])

# Domain grid
Nx, Ny = grid_len, grid_len
grid_Nsites = Nx * Ny

# Set the exchange energy
J_energy = np.diag(np.ones(Nx - 1), k=1) + np.diag(np.ones(Nx - 1), k=-1)
J_energy = np.kron(J_energy, np.eye(Nx)) + np.kron(np.eye(Nx), J_energy)
# J_energy = np.zeros((Nsites, Nsites))
# J_energy += np.diag(np.ones(Nsites - 1), k=1) + np.diag(np.ones(Nsites - 1), k=-1)
# J_energy += np.diag(np.ones(Nsites - Nx), k=Nx) + np.diag(np.ones(Nsites - Nx), k=-Nx)

# Temperature
T = 0

# Generating function of the random field
def f_gf(Nsites: int) -> float:
    R = 1.
    return np.random.randn(Nsites) * R


# Magnetic field axis
b_N = 101
b_ax = np.linspace(-3, 3, b_N)

# Number of steps per cycle
steps_number = 4000

f = f_gf(grid_Nsites)
m_up, s_up, m_down, s_down = hystersis_cycle_athermal(J_energy, f, b_ax, steps_number)

# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2)
# ax1.imshow(s_up[52].reshape(Nx, Ny), vmin=0, vmax=1)
# ax2.imshow(s_up[53].reshape(Nx, Ny), vmin=0, vmax=1)
# ax3.imshow(s_up[54].reshape(Nx, Ny), vmin=0, vmax=1)
# ax4.imshow(s_up[55].reshape(Nx, Ny), vmin=0, vmax=1)

# Check convergence
plt.plot(range(steps_number), m_up.T)

fig, ax = plt.subplots()
# ax.plot(b_ax, m_up[:, -1], ".-")
# ax.plot(b_ax, m_down[:, -1], ".-")
# ax.set_xlabel(r"$b/J$")
# ax.set_ylabel(r"$m/M_s$")

f_sn = 19 # 300  # Number of realization
m_up_s, m_down_s = disorder_averged_hysteresis_cycle_athermal(
    J_energy, f_gf, f_sn, b_ax, steps_number
)

mc_0_pi = 0.7
mc_pi_N = 0.9

m_up_s = np.append(m_up_s, m_up[:, [-1]].T, axis=0)
m_down_s = np.append(m_down_s, m_down[:, [-1]].T, axis=0)

m_up_mean = m_up_s.mean(axis=0)
m_down_mean = m_down_s.mean(axis=0)

# bins = [-1.1, -mc_pi_N, -mc_0_pi, mc_0_pi, mc_pi_N, 1.1]  # This one is for a 5 column plot
bins = 20

m_up_hist = np.histogram(m_up_s.flatten(), density = True, bins=bins)
m_down_hist = np.histogram(m_down_s.flatten(), density = True, bins=bins)

bar_colors = ([['C7'] * 1 , ['C0'] * 2, ['C3'] * 14, ['C0'] * 2, ['C7'] * 1])
bar_colors = [item for sublist in bar_colors for item in sublist]

middle_plots_idx = [67, 69, 71, 73]


fig = plt.figure( figsize = (10, 10))

gs = GridSpec(3, 4, width_ratios=[1, 1, 1, 1], height_ratios=[2, 1, 2])

ax_t = fig.add_subplot(gs[0, :])

ax_m1 = fig.add_subplot(gs[1, 0])
ax_m2 = fig.add_subplot(gs[1, 1])
ax_m3 = fig.add_subplot(gs[1, 2])
ax_m4 = fig.add_subplot(gs[1, 3])

# ax_b1 = fig.add_subplot(gs[2, :2])
# ax_b2 = fig.add_subplot(gs[2, 2:])


##### TOP
def generate_top() -> None:
    # We show only 20 curves, otherwise it gets too crowded
    for i in range(19):
        # blue curves
        ax_t.plot(b_ax, m_up_s[i, :], ".-", color="C0", alpha=0.5)
        ax_t.plot(b_ax, m_down_s[i, :], ".-", color="C0", alpha=0.5)

    # Orange curve
    ax_t.plot(b_ax, m_up_s[-1, :], ".-", color="orange")
    ax_t.plot(b_ax, m_down_s[-1, :], ".-", color="orange")

    # jump points
    orange_points_up = m_up_s[-1, :]
    orange_points__down = m_up_s[-1, :]

    print("orange_points_up", orange_points_up)
    print(len(orange_points_up))
    # print(orange_points__down)

    print("np.diff(orange_points_up)", np.diff(orange_points_up))
    # print(np.diff(orange_points__down))

    convert_into_audio(np.diff(orange_points_up), "output.wav")

    ax_t.set_xlabel(r"$b / J$")
    ax_t.set_ylabel(r"$\langle S \rangle $")

    ax_t.hlines([mc_pi_N, mc_0_pi, -mc_0_pi, -mc_pi_N], -3, 3, colors="k")

    # ax_t.text(-3, 1.0, "N")
    # ax_t.text(-3, 0.8, "$\pi$")
    # ax_t.text(-3, 0.0, "0")
    # ax_t.text(-3, -0.8, "$\pi$")
    # ax_t.text(-3, -1.0, "N")

    ax_t.scatter(
        b_ax[middle_plots_idx], m_up[:, -1][middle_plots_idx], color="C3", zorder=20
    )


generate_top()


#### INSET
def gen_green_graph(ax_t_inset: plt.Axes) -> None:
    axins = ax_t_inset.inset_axes((0.8, 0, 0.2, 0.7))
    axins.plot(b_ax, m_up_mean, ".-", color="C2")
    axins.plot(b_ax, m_down_mean, ".-", color="C2")
    axins.set_xlabel(r"$b / J$")
    axins.set_ylabel(r"$\langle S \rangle$")

    axins.hlines([mc_pi_N, mc_0_pi, -mc_0_pi, -mc_pi_N], -3, 3, colors="k")

    axins.text(-3, 1.0, "N")
    axins.text(-3, 0.8, "$\pi$")
    axins.text(-3, 0.0, "0")
    axins.text(-3, -0.8, "$\pi$")
    axins.text(-3, -1.0, "N")


# gen_green_graph(ax_t)

##### MIDDLE

ax_m1.imshow(s_up[middle_plots_idx[0] - 1].reshape(Nx, Ny), vmin=0, vmax=1)
ax_m2.imshow(s_up[middle_plots_idx[1] - 1].reshape(Nx, Ny), vmin=0, vmax=1)
ax_m3.imshow(s_up[middle_plots_idx[2] - 1].reshape(Nx, Ny), vmin=0, vmax=1)
ax_m4.imshow(s_up[middle_plots_idx[3] - 1].reshape(Nx, Ny), vmin=0, vmax=1)

for graph in [ax_m1, ax_m2, ax_m3, ax_m4]:
    graph.set_xticks(np.arange(10) - 0.5, labels=[])
    graph.set_yticks(np.arange(10) - 0.5, labels=[])
    graph.grid(visible=True, which="major", axis="both")

##### BOTTOM

# # bins = [-1.1, -mc_pi_N, -mc_0_pi, mc_0_pi, mc_pi_N, 1.1]
# bins = 30

# ax_b1.bar(m_up_hist[1][:-1], m_up_hist[0], width=0.1, align='edge', color = bar_colors)
# ax_b1.vlines([mc_pi_N, mc_0_pi, -mc_0_pi, -mc_pi_N], 0, 3, colors="k")

# ax_b1.text(1.0, 1, "N")
# ax_b1.text(0.8, 1, "$\pi$")
# ax_b1.text(0.0, 1, "0")
# ax_b1.text(-0.8, 1, "$\pi$")
# ax_b1.text(-1.0, 1, "N")
# ax_b1.set_xlabel("$m$")

# ax_b2.bar(m_down_hist[1][:-1], m_down_hist[0], width=0.1, align='edge', color = bar_colors)
# ax_b2.vlines([mc_pi_N, mc_0_pi, -mc_0_pi, -mc_pi_N], 0, 3, colors="k")

# ax_b2.text(1.0, 1, "N")
# ax_b2.text(0.8, 1, "$\pi$")
# ax_b2.text(0.0, 1, "0")
# ax_b2.text(-0.8, 1, "$\pi$")
# ax_b2.text(-1.0, 1, "N")
# ax_b2.set_xlabel("$m$")

gs.tight_layout(fig)

plt.savefig(f"ising_figure_{Nx}_{Ny}.pdf")

# bins=20
# hist=np.histogram([-1,1,-1,1,0.9,0.89,0.91], density = True, bins=bins)
# plt.bar(hist[1][:-1], hist[0], width=0.1, align='edge', color = bar_colors)
