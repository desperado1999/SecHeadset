import numpy as np
import scipy.fftpack
import scipy.signal
import soundfile as sf
from utils.parameters import *

EPS=1e-6

class noise_object:
    def __init__(self, size, dtype=float):
        self.data = np.zeros((size), dtype=dtype)
        self.idx = 0
        self.finish = False
        self.max_size = size

    def update(self, input_data):
        if(self.idx + len(input_data) > self.max_size):
            self.data[self.idx:] = input_data[:self.max_size - self.idx]
            self.finish = True
            self.idx = self.max_size
        else:
            self.data[self.idx : self.idx + len(input_data)] = input_data
            self.idx += len(input_data)
            if(self.idx == self.max_size):
                self.finish= True

    def __len__(self):
        return self.idx

def normalize_audio(audiowave):
    audiowave = audiowave - np.mean(audiowave)
    audiowave = 0.8 * audiowave / np.amax(np.abs(audiowave))
    return audiowave

def mix_to_target_snr(audio,noise,snr):
    current_snr = calculate_snr(audio, noise)
    factor = np.sqrt((10**(-snr/10))/(10**(-current_snr/10)))
    return audio + factor * noise

def denoise_according_to_snr(noised_audio, noise, snr):
    noise_power = np.mean(noise ** 2)
    noised_audio_power = np.mean(noised_audio ** 2)
    k = np.sqrt(noise_power * (10**(snr/10)+1) / noised_audio_power)

    noised_audio = k * noised_audio

    # Denoise method, such as directly substration or spectral substraction
    denoised_audio = noised_audio - noise

    return denoised_audio

def calculate_snr(audio, noise):
    audio_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10(audio_power / (noise_power + EPS) + EPS)

def generate_multiband_factor(fs=16000,
                               n_fft=1024,
                               n_band=5
                               ):
    # critical_freqs = [fs/(2**i) for i in range(n_band+1, 0, -1)]
    # critical_freqs.insert(0, 0)

    critical_freqs = [0, fs/64, fs/32, fs/16, fs/8, fs/4, fs/2]

    fft_freqs = scipy.fft.fftfreq(n_fft, 1/fs)
    fft_freqs = fft_freqs[:len(fft_freqs) // 2 + 1]
    closest_indices = [
        min(range(len(fft_freqs)), key=lambda i: abs(fft_freqs[i] - freq))
        for freq in critical_freqs
    ]

    mulitband_factor = np.ones(len(fft_freqs))

    mulitband_factor[closest_indices[0] : closest_indices[1]] = 1
    mulitband_factor[closest_indices[1] : closest_indices[2]] = 1
    mulitband_factor[closest_indices[2] : closest_indices[3]] = 1
    mulitband_factor[closest_indices[3] : closest_indices[4]] = 1
    mulitband_factor[closest_indices[4] : closest_indices[5]] = 1
    mulitband_factor[closest_indices[5] : ] = 1

    return np.expand_dims(mulitband_factor, 1)

def adaptive_spectral_substraction(noisy,
                                  noise, 
                                  fs,
                                  fr=None):
    assert len(noisy) == len(noise), 'The length of the inputs must be the same!!!'
    # assert fs == RATE, 'The sample rate of the inputs must be 16kHz!!!'
    scaling = SB_SCALING

    # Power Balance
    p_n = np.mean(noise * noise)
    p_m = np.mean(noisy * noisy)
    noisy = noisy * np.sqrt(p_n) / np.sqrt(p_m)

    edge_freq = 2048
    lpf = scipy.signal.firwin(17, edge_freq*2/fs, pass_zero='lowpass')
    hpf = scipy.signal.firwin(17, edge_freq*2/fs, pass_zero='highpass')

    noise_l = scipy.signal.lfilter(lpf, 1, noise)
    noise_l = noise_l * 2
    noisy_l = scipy.signal.lfilter(lpf, 1, noisy)

    h_nfft = SB_N_FFT_H
    l_nfft = SB_N_FFT_L
    h_overlap = SB_OVERLAP_H
    l_overlap = SB_OVERLAP_L

    [f, _, spec_noise_l] = scipy.signal.stft(noise_l, 
                                           fs=fs,
                                           window='hann',
                                           nperseg=l_nfft,
                                           noverlap=l_overlap,
                                           nfft=l_nfft,
                                           scaling=scaling,
                                           return_onesided=True
                                           )
    [_, _, spec_noisy_l] = scipy.signal.stft(noisy_l, 
                                           fs=fs,
                                           window='hann',
                                           nperseg=l_nfft,
                                           noverlap=l_overlap,
                                           nfft=l_nfft,
                                           scaling=scaling,
                                           return_onesided=True
                                           )
    if fr is not None:
        current_fr = fr(abs(f))
        current_fr = np.expand_dims(current_fr, 1)
        spec_noise_l = spec_noise_l * current_fr


    noise_h = scipy.signal.lfilter(hpf, 1, noise)
    noise_h = noise_h * 1
    # noise_h = noise_h * 1.8

    noisy_h = scipy.signal.lfilter(hpf, 1, noisy)

    [f, _, spec_noise_h] = scipy.signal.stft(noise_h, 
                                           fs=fs,
                                           window='hann',
                                           nperseg=h_nfft,
                                           noverlap=h_overlap,
                                           nfft=h_nfft,
                                           scaling=scaling,
                                           return_onesided=True
                                           )
    [_, _, spec_noisy_h] = scipy.signal.stft(noisy_h, 
                                           fs=fs,
                                           window='hann',
                                           nperseg=h_nfft,
                                           noverlap=h_overlap,
                                           nfft=h_nfft,
                                           scaling=scaling,
                                           return_onesided=True
                                           ) 

    if fr is not None:
        current_fr = fr(abs(f))
        current_fr = np.expand_dims(current_fr, 1)
        spec_noise_h = spec_noise_h * current_fr

    spec_substraction_filter = 1 - (np.abs(spec_noise_l) / (np.abs(spec_noisy_l) + 1e-10))
    spec_substraction_filter[spec_substraction_filter < 0] = 0.01
    denoised_spec_l = spec_substraction_filter * spec_noisy_l

    spec_substraction_filter = 1 - (np.abs(spec_noise_h) / (np.abs(spec_noisy_h) + 1e-10))
    spec_substraction_filter[spec_substraction_filter < 0] = 0.01
    denoised_spec_h = spec_substraction_filter * spec_noisy_h

    [_, denoised_result_l] = scipy.signal.istft(denoised_spec_l,
                                              fs=fs,
                                              window='hann',
                                              nperseg=l_nfft,
                                              noverlap=l_overlap,
                                              nfft=l_nfft,
                                              scaling=scaling,
                                              input_onesided=True
                                              )
    [_, denoised_result_h] = scipy.signal.istft(denoised_spec_h,
                                              fs=fs,
                                              window='hann',
                                              nperseg=h_nfft,
                                              noverlap=h_overlap,
                                              nfft=h_nfft,
                                              scaling=scaling,
                                              input_onesided=True
                                              )

    min_len = min(len(denoised_result_h), len(denoised_result_l))
    denoised_result = denoised_result_h[:min_len] + denoised_result_l[:min_len]

    return denoised_result

def mb_stft(input, fs, fr=None):
    assert fs==RATE, 'The sample rate does not match!!!'
    n_band = SB_MB_N_BAND
    n_fft = SB_MB_N_FFT
    scaling = SB_SCALING
    overlap_factor = SB_MB_OVERLAP_FACTOR

    result = [None] * (n_band + 1)

    h_data = input
    for i in range(n_band):
        
        # Filter
        l_data = np.zeros(len(h_data))
        ## For l_data
        for j in range(len(h_data)-1, 0, -1):
            l_data[j] = (h_data[j-1] + h_data[j]) / np.sqrt(2)

        ## For h_data
        for j in range(len(h_data)-1, 0, -1):
            h_data[j] = (h_data[j] - h_data[j-1]) / np.sqrt(2)

        # Subsample
        h_data = h_data[0::2]
        l_data = l_data[0::2]

        # Calculate coefficient for h_data
        temp_nfft = int(n_fft / (2**i))
        
        [f, _, l_coeff] = scipy.signal.stft(l_data, 
                                            fs=fs,
                                            window='hann',
                                            nperseg=temp_nfft,
                                            noverlap=int(temp_nfft*overlap_factor),
                                            nfft=temp_nfft,
                                            scaling=scaling,
                                            return_onesided=True
                                            )

        result[i] = l_coeff

    [_, _, h_coeff] = scipy.signal.stft(h_data, 
                                        fs=fs,
                                        window='hann',
                                        nperseg=temp_nfft,
                                        noverlap=int(temp_nfft*overlap_factor),
                                        nfft=temp_nfft,
                                        scaling=scaling,
                                        return_onesided=True
                                        )

    result[i+1] = h_coeff
    
    return result

def mb_istft(input, fs):
    assert fs==RATE, 'The sample rate does not match!!!'
    n_band = SB_MB_N_BAND
    n_fft = SB_MB_N_FFT
    scaling = SB_SCALING
    overlap_factor = SB_MB_OVERLAP_FACTOR

    temp_nfft = int(n_fft / 2**(n_band - 1))
    [_, h_result] = scipy.signal.istft(input[n_band],
                                    fs=fs,
                                    window='hann',
                                    nperseg=temp_nfft,
                                    noverlap=int(temp_nfft*overlap_factor),
                                    nfft=temp_nfft,
                                    scaling=scaling,
                                    input_onesided=True
                                    )

    for i in range(n_band-1, -1, -1):
        temp_nfft = int(n_fft / 2**i)
        [_, l_result] = scipy.signal.istft(input[i],
                                        fs=fs,
                                        window='hann',
                                        nperseg=temp_nfft,
                                        noverlap=int(temp_nfft*overlap_factor),
                                        nfft=temp_nfft,
                                        scaling=scaling,
                                        input_onesided=True
                                        )

        # Oversample
        h_result = np.repeat(h_result, 2)
        h_result[1::2] = 0

        l_result = np.repeat(l_result, 2)
        l_result[1::2] = 0

        # Reverse filter
        ## For l_data
        for j in range(len(l_result) - 1):
            l_result[j] = (l_result[j] + l_result[j+1]) / np.sqrt(2)
        ## For h_data
        for j in range(len(h_result) - 1):
            h_result[j] = (h_result[j] - h_result[j+1]) / np.sqrt(2)

        h_result = l_result + h_result

    return np.real(h_result)

def mb_spectral_substraction(noisy,
                          noise,
                          fs,
                          n_band,
                          fr=None):

    assert len(noisy) == len(noise), 'The length of the inputs must be the same!!!'
    if fs != 16000:
        print(f'The sample rate of the input audios is {fs} instead of 16 kHz. The parameters need finetune for better performance.')

    # Power Balance
    p_n = np.mean(noise * noise)
    p_m = np.mean(noisy * noisy)
    noisy = noisy * np.sqrt(p_n) / np.sqrt(p_m)

    mb_spec_noise = mb_stft(noise, fs, fr)
    mb_spec_noisy = mb_stft(noisy, fs)

    factor = np.ones(n_band+ 1)
    for i in range(len(mb_spec_noise)):
        spec_substraction_filter = 1 - (
            np.abs(mb_spec_noise[i] * factor[i]) / (np.abs(mb_spec_noisy[i]) + 1e-9)
        )
        spec_substraction_filter[spec_substraction_filter < 0] = 0.01
        mb_spec_noisy[i] = spec_substraction_filter * mb_spec_noisy[i]

    result = mb_istft(mb_spec_noisy, fs)

    return result
