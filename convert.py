import numpy as np
import scipy.io.wavfile as wavfile
from scipy.fft import ifft
import matplotlib.pyplot as plt


def convert_into_audio(M_t: list[float] | np.ndarray, file_name: str) -> None:
    M_t = np.array(M_t)

    F_k = np.fft.fft(M_t)
    M_t_reconstructed = ifft(F_k).real

    # -1,1 normalization
    M_t_reconstructed = M_t_reconstructed / np.max(np.abs(M_t_reconstructed))
    sample_rate = 44100
    wavfile.write(file_name, sample_rate, M_t_reconstructed.astype(np.float32))
