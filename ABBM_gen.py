import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

for idx in range(300):
    np.random.seed(idx)

    # Parameters
    Gamma = 1.0  # Damping coefficient
    k = 1.0  # Spring constant (restoring force strength)
    H_step = 1.0  # Step in the external driving field
    dt = 0.001  # Time step
    T = 30.0  # Total simulation time
    m0 = 0.0  # Initial displacement
    D = 0.1  # Amplitude of the noise
    L = 10.0  # Length of the spatial range for W(m)
    dx = 0.00001  # Sampling step for the spatial range

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

    # plt.figure(figsize=(10, 6))
    # plt.plot(
    #     time[start_from:-1],
    #     velocity[start_from:],
    #     label="Velocity $\\frac{dm}{dt}$",
    #     color="blue",
    # )
    # plt.xlabel("Time")
    # plt.ylabel("Velocity $\\frac{dm}{dt}$")
    # plt.title("Velocity of Domain Wall Displacement from T/4")
    # plt.legend()
    # plt.grid()
    # plt.show()

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
        f"data/default_values/{idx}.wav",
        sample_rate,
        (samples * MAX_INT_16_VALUE).astype(np.int16),
    )

    # # Plot velocity over time
    # plt.plot(time, velocity)
    # plt.title("Velocity vs Time")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Velocity")
    # plt.grid(True)
    # plt.show()
