import pathos as pa
import numpy as np
from scipy.io.wavfile import write
from tqdm import tqdm
from os import path, makedirs
from parameters import parameters


def create_dir_if_not_exists(directory: str) -> None:
    if not path.exists(directory):
        makedirs(directory)


config = [
    # 1, 10, 100, 1000
    # default
    {"dir_samples": "new-f-1", "Gamma": 1.0, "k": 1.0, "D": 0.1, "f": 1.0},
    {"dir_samples": "new-f-10", "Gamma": 1.0, "k": 1.0, "D": 0.1, "f": 10.0},
    {"dir_samples": "new-f-100", "Gamma": 1.0, "k": 1.0, "D": 0.1, "f": 100.0},
    {"dir_samples": "new-f-1000", "Gamma": 1.0, "k": 1.0, "D": 0.1, "f": 1000.0},
    # gamma10
    {"dir_samples": "new-f-1-gamma10", "Gamma": 10.0, "k": 1.0, "D": 0.1, "f": 1.0},
    {"dir_samples": "new-f-10-gamma10", "Gamma": 10.0, "k": 1.0, "D": 0.1, "f": 10.0},
    {"dir_samples": "new-f-100-gamma10", "Gamma": 10.0, "k": 1.0, "D": 0.1, "f": 100.0},
    {"dir_samples": "new-f-1000", "Gamma": 10.0, "k": 1.0, "D": 0.1, "f": 1000.0},
    # k10
    {"dir_samples": "new-f-1-k10", "Gamma": 1.0, "k": 10.0, "D": 0.1, "f": 1.0},
    {"dir_samples": "new-f-10-k10", "Gamma": 1.0, "k": 10.0, "D": 0.1, "f": 10.0},
    {"dir_samples": "new-f-100-k10", "Gamma": 1.0, "k": 10.0, "D": 0.1, "f": 100.0},
    {"dir_samples": "new-f-1000-k10", "Gamma": 1.0, "k": 10.0, "D": 0.1, "f": 1000.0},
    # noise 05
    {"dir_samples": "new-f-1-noise05", "Gamma": 1.0, "k": 1.0, "D": 0.5, "f": 1.0},
    {"dir_samples": "new-f-10-noise05", "Gamma": 1.0, "k": 1.0, "D": 0.5, "f": 10.0},
    {"dir_samples": "new-f-100-noise05", "Gamma": 1.0, "k": 1.0, "D": 0.5, "f": 100.0},
    {"dir_samples": "new-f-1000-noise05", "Gamma": 1.0, "k": 1.0, "D": 0.5, "f": 1000.0},
    # # default
    # {"dir_samples": "f1-default", "Gamma": 1.0, "k": 1.0, "D": 0.1, "f": 1.0},
    # {"dir_samples": "f10-default", "Gamma": 1.0, "k": 1.0, "D": 0.1, "f": 10.0},
    # {"dir_samples": "f15-default", "Gamma": 1.0, "k": 1.0, "D": 0.1, "f": 15.0},
    # {"dir_samples": "f30-default", "Gamma": 1.0, "k": 1.0, "D": 0.1, "f": 30.0},
    # # gamma10
    # {"dir_samples": "f1-gamma10", "Gamma": 10.0, "k": 1.0, "D": 0.1, "f": 1.0},
    # {"dir_samples": "f10-gamma10", "Gamma": 10.0, "k": 1.0, "D": 0.1, "f": 10.0},
    # {"dir_samples": "f15-gamma10", "Gamma": 10.0, "k": 1.0, "D": 0.1, "f": 15.0},
    # {"dir_samples": "f30-gamma10", "Gamma": 10.0, "k": 1.0, "D": 0.1, "f": 30.0},
    # # k10
    # {"dir_samples": "f1-k10", "Gamma": 1.0, "k": 10.0, "D": 0.1, "f": 1.0},
    # {"dir_samples": "f10-k10", "Gamma": 1.0, "k": 10.0, "D": 0.1, "f": 10.0},
    # {"dir_samples": "f15-k10", "Gamma": 1.0, "k": 10.0, "D": 0.1, "f": 15.0},
    # {"dir_samples": "f30-k10", "Gamma": 1.0, "k": 10.0, "D": 0.1, "f": 30.0},
    # # noise 05
    # {"dir_samples": "f1-noise05", "Gamma": 1.0, "k": 1.0, "D": 0.5, "f": 1.0},
    # {"dir_samples": "f10-noise05", "Gamma": 1.0, "k": 1.0, "D": 0.5, "f": 10.0},
    # {"dir_samples": "f15-noise05", "Gamma": 1.0, "k": 1.0, "D": 0.5, "f": 15.0},
    # {"dir_samples": "f30-noise05", "Gamma": 1.0, "k": 1.0, "D": 0.5, "f": 30.0},
]

params = {}

for fparams in config:
    create_dir_if_not_exists(f"data/{fparams['dir_samples']}")

    def gen_abbm_audio(confs: dict) -> None:

        idx = confs["idx"]

        np.random.seed(idx)

        H_step, m0, dx = parameters["H_step"], parameters["m0"], parameters["dx"]
        Gamma, k, D = confs["Gamma"], confs["k"], confs["D"]
        L, dt, T = parameters["L"], parameters["dt"], parameters["T"]
        f = confs["f"]

        A = 1

        # Spatial range for W(m)
        x = np.arange(-L / 2, L / 2, dx)
        N = len(x)

        # Generate a correlated random force W(m) using a Brownian motion approach
        increments = np.random.normal(0, np.sqrt(2 * D * dx), size=N - 1)
        correlated_force = np.zeros(N)
        correlated_force[1:] = np.cumsum(
            increments
        )  # Cumulative sum generates Brownian motion

        # Time points for simulation
        # time = np.arange(-T/4, T, dt)
        time = np.arange(0, T, dt)
        H = np.zeros_like(time)

        # H[time > 0] = A * np.cos( 2 * np.pi * f time[time > T / 4])  # Step in H at T/4
        H[time > T / 4] = A * np.sin(
            2 * np.pi * f * (time[time > T / 4] - T / 4)
        )  # Step in H at T/4

        # Initialize displacement array for the ABBM model
        m = np.zeros_like(time)
        m[0] = m0

        # Simulate the dynamics of the ABBM model with the correlated force
        for i in range(1, len(time)):

            # Update interpolated force
            W_interp = np.interp(m[i - 1], x, correlated_force)

            # Compute the net force and update displacement using Euler integration
            dm = (H[i] - k * m[i - 1] + W_interp) / Gamma * dt
            m[i] = m[i - 1] + dm

        # velocity (dm/dt) over time
        velocity = np.diff(m) / dt
        start_from = int(len(time) / 4)

        time = time[start_from:-1]
        velocity = velocity[start_from:]

        ##### split funciton until here #####

        # normalize time, remove initial silence
        diff_time = time[0]
        time = [t - diff_time for t in time]

        # Normalize the signal to fit within the audio range (-1, 1)
        velocity_normalized = velocity / np.max(np.abs(velocity))

        # Audio settings
        sample_rate = 44100
        duration = time[-1]
        samples = np.interp(
            np.linspace(0, duration, int(sample_rate * duration)),
            time,
            velocity_normalized,
        )

        MAX_INT_16_VALUE = 32767
        write(
            f"data/{confs['dir_samples']}/{idx+1}.wav",
            sample_rate,
            (samples * MAX_INT_16_VALUE).astype(np.int16),
        )

    # create_dir_if_not_exists("data/samples")

    NUM_SAMPLES = int(parameters["NUM_SAMPLES"])

    # for index in tqdm(range(NUM_SAMPLES)):
    #     gen_abbm_audio(index)

    # hackfix for passing multiple conf parameters to function
    all_confs = []
    for i in range(NUM_SAMPLES):
        confs = fparams.copy()
        confs["idx"] = i
        all_confs.append(confs)

    ncpu = parameters["ncpu"]
    with pa.multiprocessing.ProcessingPool(ncpu) as p:
        res = list(tqdm(p.imap(gen_abbm_audio, all_confs), total=NUM_SAMPLES))
