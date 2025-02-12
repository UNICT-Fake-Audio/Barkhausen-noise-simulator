import pathos as pa
import numpy as np
from scipy.io.wavfile import write
from tqdm import tqdm
from os import path, makedirs
from parameters import parameters


def create_dir_if_not_exists(directory: str) -> None:
    if not path.exists(directory):
        makedirs(directory)


def gen_abbm_audio(idx: int) -> None:
    np.random.seed(idx)

    H_step, m0, dx = parameters["H_step"], parameters["m0"], parameters["dx"]
    Gamma, k, D = parameters["Gamma"], parameters["k"], parameters["D"]
    L, dt, T = parameters["L"], parameters["dt"], parameters["T"]

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
    time = np.arange(0, T, dt)
    H = np.zeros_like(time)
    H[time > T / 4] = np.cos(time[time > T / 4])  # Step in H at T/4

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

    # normalize time, remove initial silence
    diff_time = time[0]
    time = [t - diff_time for t in time]

    # Normalize the signal to fit within the audio range (-1, 1)
    velocity_normalized = velocity / np.max(np.abs(velocity))

    # Audio settings
    sample_rate = 44100  # Sampling rate (44.1 kHz)
    duration = time[-1]
    samples = np.interp(
        np.linspace(0, duration, int(sample_rate * duration)), time, velocity_normalized
    )

    # Save to audio file (wav)
    MAX_INT_16_VALUE = 32767
    write(
        f"data/samples/{idx+1}.wav",
        sample_rate,
        (samples * MAX_INT_16_VALUE).astype(np.int16),
    )


create_dir_if_not_exists("data/samples")

NUM_SAMPLES = int(parameters["NUM_SAMPLES"])

# for index in tqdm(range(NUM_SAMPLES)):
#     gen_abbm_audio(index)

ncpu = parameters["ncpu"]
with pa.multiprocessing.ProcessingPool(ncpu) as p:
    res = list(tqdm(p.imap(gen_abbm_audio, range(NUM_SAMPLES)), total=NUM_SAMPLES))
