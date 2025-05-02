import scipy
import scipy.signal
import numpy as np
import soundfile as sf

from utils.parameters import *


def generate_test_data():
    data = np.random.randint(low=0, high=2, size=(11,len(CARRIER_FREQS)-1))
    preamble  = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
    data[0] = preamble[:len(CARRIER_FREQS)-1]
    print(data)
    
    # Save to npy
    np.save('./data/temp_data.npy', data, allow_pickle=True)

    generate_modulated_wave()

def generate_modulated_wave():
    data = np.load('./data/temp_data.npy', allow_pickle=True)

    modulated_wave = np.array([])
    for i in range(data.shape[0]):
        modulated_wave = np.append(modulated_wave, encode_data(data[i]))
        modulated_wave = np.append(modulated_wave, np.zeros(CHUNK))
    
    try:
        modulated_wave = modulated_wave / np.max(np.abs(modulated_wave))
    except:
        pass
    sf.write('./test.wav', modulated_wave, RATE)


def encode_data(data):
    t = np.linspace(1, CHUNK, CHUNK)
    assert len(data) == len(CARRIER_FREQS) - 1, "The number of data does not match the number of carriers"
    modulated_data = np.zeros(CHUNK)

    # Use lowest freq value [0] as ref
    for i in range(1, len(CARRIER_FREQS)):
        phase = 2 * np.pi * np.random.randn() + 2 * np.pi * CARRIER_FREQS[i] * t / RATE
        modulated_data = modulated_data + data[i-1] * np.cos(phase) * WINDOW

    normalize_factor = np.max(np.abs(modulated_data))
    if normalize_factor > 0:
        modulated_data = modulated_data / normalize_factor

    return modulated_data

def VAD(audio_data):
    avg_amp = np.mean(np.abs(audio_data))
    
    return avg_amp >= VAD_THRESHOLD

def corr(x, y):
    assert len(x) == len(y), 'The length of the inputs do not match!!!'

    a = np.sum(x * y)
    b = np.sqrt(np.sum(x * x) * np.sum(y * y))
    
    if b==0:
        return 0

    return a/b

def corr_2d(x, y):
    assert len(x.shape) == 2, 'The inputs are not 2-dimension data!!!'
    assert x.shape == y.shape, 'The shape of the inputs do not match!!!'

    a = np.sum(x * y)
    b = np.sqrt(np.sum(x * x) * np.sum(y * y))
    
    if b==0:
        return 0

    return a/b

def corr_freq(x, y):
    """
    Correlation in frequency domain.

    ToDo:
        1. Pre-define some parameters to reduce computational overhead.
            1. fft_y
    """
    assert len(x) == len(y), 'The length of the inputs do not match!!!'

    min_freq = 4000
    fft_freq = scipy.fft.fftfreq(len(x), 1.0/RATE)

    min_freq_index = np.argmin(np.abs(fft_freq - min_freq))
    max_freq_index = int(len(x)/2) + 1

    fft_x = scipy.fft.fft(x)
    fft_y = scipy.fft.fft(y)

    fft_x = np.abs(fft_x[min_freq_index:max_freq_index])
    fft_y = np.abs(fft_y[min_freq_index:max_freq_index])

    return corr(fft_x, fft_y)

def generate_preamble(fs=RATE,
                      tone_len=TONE_LEN,
                      tone_freq=TONE_FREQ,
                      chirp_len=CHIRP_LEN,
                      chirp_start_freq=CHIRP_START_FREQ,
                      chirp_end_freq=CHIRP_END_FREQ
                      ):
    blank_len = tone_len
    tone_t = np.linspace(1/fs, tone_len, int((tone_len * fs)))
    tone = np.sin(2 * np.pi * tone_freq * tone_t)
    blank = np.zeros(int(blank_len * fs))

    chirp_t = np.linspace(1/fs, chirp_len, int((chirp_len * fs)))
    chirp1 = scipy.signal.chirp(chirp_t, chirp_start_freq, chirp_len, chirp_end_freq)
    chirp2 = scipy.signal.chirp(chirp_t, chirp_end_freq, chirp_len, chirp_start_freq)

    # Chirp = [chirp1, chirp2]
    chirp = np.concatenate((chirp1, chirp2))
    chirp = 0.8 * chirp / np.max(np.abs(chirp)) 
    sf.write(CHIRP_PATTERN_PATH, chirp, fs)


    # preamble = [tone, blank, chirp1, chirp2]
    preamble = np.concatenate((tone, blank, chirp1, chirp2))
    
    # Make the length of CHUNK preamble the integral multiple of CHUNK
    if len(preamble) % CHUNK != 0:
        preamble = np.concatenate((preamble, np.zeros((CHUNK - len(preamble) % CHUNK))))
    preamble = 0.8 * preamble / np.max(np.abs(preamble)) 

    sf.write(PREAMBLE_PATH, preamble, fs)
    return preamble, fs

def resample(input_data, origin_fs, target_fs):
    return scipy.signal.resample_poly(input_data, target_fs, origin_fs)

# Wait for deletion
def correlation_with_normalization(x, y):
    """
    Cross-correlation of two 1-d arrays
    Inputs:
        x: the first array
        y: the second array
    output:
        normalized_corr: the normalized correlation results
    
    Requirements:
        1. Len (x) >= Len (y)
        2. No complex data

    Tips:
        1. The normalized here means: z[k] = sum(x_i * y_j) / (l2_norm(x_i) * l2_norm(y_i))
    """
    normalized_corr = np.zeros(len(x))
    x = np.pad(x, (len(y) - 1, 0), mode='constant', constant_values=0)

    y_mul_sum = np.sum(y * y)

    for end_point in range(len(y), len(x)):
        current_data = x[end_point - len(y) : end_point]
        normalized_corr[end_point - len(y)] = np.sum(current_data * y) / np.sqrt(
            (y_mul_sum * np.sum(current_data * current_data))
        )
    return normalized_corr 

def decoding_from_spectrum(fft_data, calculate_error=0):
    """
    Demodulated data from fft result of a CHUNK of audio data
    Inputs:
        fft_data: np.ndarray
            fft result of a chunk of audio data (np.abs(scipy.fft.fft(data[:CHUNK] * HAMMING_WINDOW)))
    Outputs:
        modulated_data: list
            list of the demodulated data
    """

    compensated_fft_data = fft_data[CARRIER_INDICES] / FREQUENCY_RESPONSE_DATA_LIST[CARRIER_INDICES]

    # Calcualte energy
    compensated_fft_data = compensated_fft_data * compensated_fft_data

    # Use the lowest freq value as [0] ref
    ref = compensated_fft_data[0]
    threshold = 10

    modulated_data = [
        1 if (value > ref * threshold) & (value > 50) else 0
        for value in compensated_fft_data[1:]
    ]

    if calculate_error == 1:
        compensated_fft_data = compensated_fft_data[:-1]
        min_value = np.min(compensated_fft_data)
        max_value = np.max(compensated_fft_data)

        flipped_modualated_data = 1 - np.array(modulated_data)

        min_error = compensated_fft_data - min_value
        max_error = max_value - compensated_fft_data

        error = np.sum(min_error * flipped_modualated_data) + np.sum(
            max_error * modulated_data
        )

        return modulated_data, error

    return modulated_data, modulated_data

if __name__ == '__main__':
    generate_test_data()
