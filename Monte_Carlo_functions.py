import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

plt.ion()

def random_spin_field(N, M):
    return np.random.choice([-1, 1], size=(N, M))

def display_spin_field(field):
    return Image.fromarray(np.uint8((field + 1) * 0.5 * 255))

def _ising_update(field, n, m, beta=0.4, h = 0.0):
    N, M = field.shape
    total = (
        field[(n + 1) % N, m] +
        field[(n - 1) % N, m] +
        field[n, (m + 1) % M] +
        field[n, (m - 1) % M]
    )
    dE = 2 * field[n, m] * (total+h)
    if dE <= 0 or np.exp(-dE * beta) > np.random.rand():
        field[n, m] *= -1

def ising_step(field, beta=0.4, h = 0.0):
    N, M = field.shape
    for n_offset in range(2):
        for m_offset in range(2):
            for n in range(n_offset, N, 2):
                for m in range(m_offset, M, 2):
                    _ising_update(field, n, m, beta,h)
    return field

def magnetization(field):
    return np.mean(field)

def two_point_correlation_r_connected(field, r):
    shifted = np.roll(field, -r, axis=1)
    raw_corr = np.mean(field * shifted)
    mean_spin = np.mean(field)
    return raw_corr - mean_spin**2

def two_point_correlation_r(field, r):
    # Shift the field by r sites to the right
    shifted_field = np.roll(field, -r, axis=1)
    corr = field * shifted_field
    return np.mean(corr)

def compute_two_point_r_profile(field, r_max):
    G = [two_point_correlation_r_connected(field, r) for r in range(1, r_max+1)]
    return np.array(G)

def two_point_correlation_t_connected(field, r): #not tested
    shifted = np.roll(field, -r, axis=0)
    raw_corr = np.mean(field * shifted)
    mean_spin = np.mean(field)
    return raw_corr - mean_spin**2

def two_point_correlation_t(field, r): #not tested
    shifted_field = np.roll(field, -r, axis=0)
    corr = field * shifted_field
    return np.mean(corr)

def compute_two_point_t_profile(field, r_max): #not tested
    G = [two_point_correlation_t_connected(field, r) for r in range(1, r_max+1)]
    return np.array(G)


def cft_prediction(r_values, A=1.0):
    Delta_sigma = 1/8
    return A / (r_values ** (2* Delta_sigma))

def Thermal_cft_t_prediction(r_values, Beta, A=1.0):
    Delta_sigma = 1/8
    factor = (np.pi / Beta) / np.sin(np.pi * r_values / Beta)
    return A * factor**(2 * Delta_sigma)

def Thermal_cft_r_prediction(r_values, Beta, A=1.0):
    Delta_sigma = 1/8
    factor = (np.pi / Beta) / np.sinh(np.pi * r_values / Beta)
    return A * factor**(2 * Delta_sigma)

def Thermal_cft_t_prediction_eps(r_values, Beta, A=1.0):
    Delta_sigma = 1
    factor = (np.pi / Beta) / np.sin(np.pi * r_values / Beta)
    return A * factor**(2 * Delta_sigma)

def Thermal_cft_r_prediction_eps(r_values, Beta, A=1.0):
    Delta_sigma = 1
    factor = (np.pi / Beta) / np.sinh(np.pi * r_values / Beta)
    return A * factor**(2 * Delta_sigma)

def Thermal_cft_t_prediction_eps(r_values, Beta, A=1.0):
    Delta_sigma = 1
    factor = (np.pi / Beta) / np.sin(np.pi * r_values / Beta)
    return A * factor**(2 * Delta_sigma)

def plot_two_point_sigma(field, r_max):
    tp = compute_two_point_r_profile(field, r_max)
    A_fit = tp[0]  # fixing normalization
    G_cft = cft_prediction(np.arange(1, r_max + 1), A_fit)
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, r_max + 1), tp, 'o', label='Monte Carlo', markersize=4)
    plt.plot(np.arange(1, r_max + 1), G_cft, '-', label='CFT ($\sim r^{-1/4}$)')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(r' Distance $r$')
    plt.ylabel(r'$\langle \sigma(0) \sigma(r) \rangle$')
    plt.title('Comparison with CFT correlator after {i} steps'.format(i=i))
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(f"Ising_correlator_step_{i}.png")
    plt.close()
    return

def plot_Thermal_two_point_r_sigma(field, r_max,Beta ,filename, step_i = 0):
    tp = compute_two_point_r_profile(field, r_max)
    A_fit = tp[0]  # fixing normalization
    G_cft = Thermal_cft_r_prediction(np.arange(1, r_max + 1), Beta ,A_fit)
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, r_max + 1), tp, 'o', label='Monte Carlo', markersize=4)
    plt.plot(np.arange(1, r_max + 1), G_cft, '-', label='Thermal CFT')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(r' Distance $r$')
    plt.ylabel(r'$\langle \sigma(r) \sigma(0) \rangle$')
    plt.title('Comparison with CFT correlator after {i} steps and $1/T$ = {Beta}'.format(i= step_i,Beta = Beta))
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(filename)
    plt.close()
    return

def plot_Thermal_two_point_t_sigma(field, r_max,Beta,filename, step_i = 0 ):
    tp = compute_two_point_t_profile(field, r_max)
    A_fit = tp[0]  # fixing normalization
    G_cft = Thermal_cft_t_prediction(np.arange(1, r_max +1), Beta ,A_fit)
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, r_max + 1), tp, 'o', label='Monte Carlo', markersize=4)
    plt.plot(np.arange(1, r_max + 1), G_cft, '-', label='Thermal CFT')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(r' Distance $\tau$')
    plt.ylabel(r'$\langle \sigma(\tau) \sigma(0) \rangle$')
    plt.title('Comparison with CFT correlator after {i} steps and $1/T$ = {Beta}'.format(i= step_i,Beta = Beta))
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(filename)
    plt.close()
    return

def export_two_point_data_txt(G,  filename='two_point_profile.txt'):
    r_values = np.arange(1, len(G) + 1)
    with open(filename, 'w') as f:
        for r, g in zip(r_values, G):
            f.write(f"{r:4d} {g:15.8f}\n")



def plot_magnetization(magnetization):
    steps = np.arange(len(magnetization))
    plt.figure(figsize=(8, 6))
    plt.plot(steps, magnetization, marker='o', label="Magnetization")
    plt.axhline(np.mean(magnetization), color='red', linestyle='--', label="Mean")
    plt.xlabel('Steps')
    plt.ylabel(r'$\langle \sigma \rangle$')
    plt.title('Magnetization as a function of steps')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig("Ising_magnetization.png")
    plt.close()

def export_field_to_tex(field, filename="field_config.tex"):
    with open(filename, 'w') as f:
        f.write("\\begin{bmatrix}\n")
        for row in field:
            row_str = " & ".join(str(int(s)) for s in row)
            f.write(row_str + " \\\\\n")
        f.write("\\end{bmatrix}\n")

def balanced_spin_field(N, M):
    total_sites = N * M
    spins = np.array([1] * (total_sites // 2) + [-1] * (total_sites - total_sites // 2))
    np.random.shuffle(spins)
    return spins.reshape((N, M))


def compute_epsilon_field(field):
    N, M = field.shape

    neighbors_sum = (
        np.roll(field, 1, axis=0) +
        np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) +
        np.roll(field, -1, axis=1)
    )
    epsilon = -field * neighbors_sum
    return epsilon

def energy_expectation_value(field):
    return np.mean(compute_epsilon_field(field))

def compute_two_point_epsilon_r_profile(field, r_max):
    epsilon = compute_epsilon_field(field)
    mean_eps = np.mean(epsilon)
    G = []
    for r in range(1, r_max+1):
        shifted = np.roll(epsilon, -r, axis=1)
        corr = np.mean(epsilon * shifted) - mean_eps**2
        G.append(corr)
    return np.array(G)

def compute_two_point_epsilon_t_profile(field, t_max):
    epsilon = compute_epsilon_field(field)
    mean_eps = np.mean(epsilon)
    G = []
    for t in range(1, t_max+1):
        shifted = np.roll(epsilon, -t, axis=0)
        corr = np.mean(epsilon * shifted) - mean_eps**2
        G.append(corr)
    return np.array(G)

def plot_Thermal_two_point_r_epsilon(field, r_max,Beta ,filename, step_i = 0):
    tp =   compute_two_point_epsilon_r_profile(field, r_max)
    A_fit = tp[0]  # fixing normalization
    G_cft = Thermal_cft_r_prediction_eps(np.arange(1, r_max + 1), Beta ,A_fit)
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, r_max + 1), tp, 'o', label='Monte Carlo', markersize=4)
    plt.plot(np.arange(1, r_max + 1), G_cft, '-', label='Thermal CFT')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(r' Distance $r$')
    plt.ylabel(r'$\langle \epsilon(r) \epsilon(0) \rangle$')
    plt.title('Comparison with CFT correlator after {i} steps and $1/T$ = {Beta}'.format(i= step_i,Beta = Beta))
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(filename)
    plt.close()
    return

def plot_Thermal_two_point_t_epsilon(field, r_max,Beta ,filename, step_i = 0):
    tp =   compute_two_point_epsilon_t_profile(field, r_max)
    A_fit = tp[0]  # fixing normalization
    G_cft = Thermal_cft_t_prediction_eps(np.arange(1, r_max + 1), Beta ,A_fit)
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, r_max + 1), tp, 'o', label='Monte Carlo', markersize=4)
    plt.plot(np.arange(1, r_max + 1), G_cft, '-', label='Thermal CFT')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(r' Distance $\tau$')
    plt.ylabel(r'$\langle \epsilon(\tau) \epsilon(0) \rangle$')
    plt.title('Comparison with CFT correlator after {i} steps and $1/T$ = {Beta}'.format(i= step_i,Beta = Beta))
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(filename)
    plt.close()
    return


