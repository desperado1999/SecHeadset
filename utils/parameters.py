import scipy.signal
import scipy
import soundfile as sf
import numpy as np

# Parameters for soundcard
SOUND_CARD_KEYWORD = 'USB'
# SOUND_CARD_KEYWORD = 'BlackHole 2ch'

# Initialize global parameters
RATE = 16000
UPDATE_WINDOW_N_FRAMES = 25
CHUNK = int(RATE/UPDATE_WINDOW_N_FRAMES)

# For preamble generation
TONE_FREQ = 3000
TONE_LEN = 3 * CHUNK / RATE
CHIRP_LEN = 6 * CHUNK / RATE
CHIRP_START_FREQ = 100
CHIRP_END_FREQ = 7000

# CHIRP_START_FREQ = 3000
# CHIRP_END_FREQ = 4000
CHIRP_PATTERN_PATH = './data/chirp_pattern.wav'
PREAMBLE_PATH = './data/preamble.wav'

# For start detection
CHUNK_STEP = int(CHUNK/8)
HAMM_WINDOW = scipy.signal.get_window('hamm', CHUNK)
HAMM_WINDOW_SCALE_FACTOR = HAMM_WINDOW.sum()

FFT_CHUNK_FREQ = scipy.fft.fftfreq(CHUNK, 1.0/RATE)

# Start Chirp Pattern
[START_PATTERN, _] = sf.read('./data/chirp_pattern.wav')

LEN_START_PATTERN = len(START_PATTERN)
     
# Spectrogram of start chirp pattern
_, _, START_PATTERN_SPEC = scipy.signal.stft(START_PATTERN,
                                window=HAMM_WINDOW,
                                nperseg=CHUNK,
                                noverlap=CHUNK-CHUNK_STEP,
                                nfft=CHUNK,
                                return_onesided=False,
                                detrend=False,
                                boundary=None,
                                # scaling='spectrum',
                                )
START_PATTERN_SPEC = abs(START_PATTERN_SPEC)

# WINDOW_TYPE='hann'
# WINDOW = scipy.signal.windows.hann(CHUNK, sym=False)
WINDOW = scipy.signal.windows.kaiser(CHUNK, beta=1)

# Parameters for VAD
VAD_THRESHOLD = 1e-4

# For data decoding
START_CARRIER_FREQ = 800
END_CARRIER_FREQ = 5600
CARRIER_STEP_FREQ = 400
CARRIER_FREQS = range(START_CARRIER_FREQ, END_CARRIER_FREQ + 1, CARRIER_STEP_FREQ)
CARRIER_INDICES = [list(FFT_CHUNK_FREQ).index(value) for value in CARRIER_FREQS]

# Get freqency response parameter
FREQUENCY_RESPONSE_DATA_DICT = np.load('./data/freq_response.npz')
FREQUENCY_RESPONSE_DATA_LIST = np.array([FREQUENCY_RESPONSE_DATA_DICT[str(int(abs(item)))] for item in FFT_CHUNK_FREQ[:int(CHUNK/2)+1]])

# For spectral subtraction
SB_MB_N_FFT = 1024
SB_MB_N_BAND = 5
SB_MB_OVERLAP_FACTOR = 3/4

SB_SCALING = 'psd'

SB_N_FFT_H = 1024
SB_OVERLAP_H = SB_N_FFT_H * 7 / 8

SB_N_FFT_L = 4096
SB_OVERLAP_L = SB_N_FFT_L * 3 / 4

SB_PADDING = True