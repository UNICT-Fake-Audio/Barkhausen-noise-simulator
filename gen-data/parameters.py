parameters = {
    "NUM_SAMPLES": 1000,  # 1000,
    "ncpu": 8,
    "H_step": 1.0,  # Step in the external driving field
    "m0": 0.0,  # Initial displacement
    "dx": 0.00001,  # Sampling step for the spatial range
    # model parameters
    "Gamma": 1.0,  # Damping coefficient
    "k": 1.0,  # Spring constant (restoring force strength)
    "D": 0.1,  # Amplitude of the noise
    "f": 1.0/6.0,  # Frequency of the probe signal (ν)
    # time/length
    "L": 10.0,  # Length of the spatial range for W(m)
    "dt": 0.001,  # Time step
    "T": 30.0,  # Total simulation time
}
# 1,10,100,1000
