import numpy as np
import scipy.signal

MAX_AMP = 0.9
FS = 16000
FREQS = np.linspace(0,8000,513)
LEN_FREQS = 513
BOUNDARY = [20, 160, 394, 670, 1000, 1420, 1900, 2450, 3120, 4000, 5100, 6600]
BOUNDARY_INDEX = np.array([1, 10, 25, 43, 64, 91, 122, 157, 200, 256, 326, 422])
LEN_BOUNDARY_INDEX = 12
FACTOR_RANGE = np.arange(0.2, 1.2, 0.1)

def update_result(noise_spec:np.ndarray, mix_spec:np.ndarray):
    spec_substraction_filter = 1 - (np.abs(noise_spec) / (np.abs(mix_spec) + 1e-30))
    denoised_spec = spec_substraction_filter * mix_spec
    return denoised_spec

def calculate_similarity(denoised_spec:np.ndarray):
    sim = np.zeros(LEN_BOUNDARY_INDEX-1)
    for i in range(LEN_BOUNDARY_INDEX-1):
        sim[i] = np.sum(
            np.abs(denoised_spec[BOUNDARY_INDEX[i] : BOUNDARY_INDEX[i + 1], :])
        )   
    return sim

def optimize_with_bs(mix_spec:np.ndarray,
                     noise_spec:np.ndarray,
                     freq_factors:np.ndarray):
    target_factor = np.ones(LEN_FREQS) * 0.2
    min_sim = 1e6 * np.ones(LEN_BOUNDARY_INDEX - 1)

    for temp_factor in FACTOR_RANGE:
        factor = np.ones(LEN_FREQS) * temp_factor
        new_noise_spec = noise_spec * np.expand_dims(factor, 1)
        new_denoised_spec = update_result(new_noise_spec, mix_spec)
        current_sim = calculate_similarity(
            denoised_spec=new_denoised_spec,
        )
        for i in range(LEN_BOUNDARY_INDEX-1):
            if current_sim[i] < min_sim[i]:
                min_sim[i] = current_sim[i]
                target_factor[BOUNDARY_INDEX[i]:BOUNDARY_INDEX[i+1]] = temp_factor

    new_target_factor = np.copy(target_factor)

    for i in range(1, 21):
        temp_factor = target_factor - 0.1 + i * 0.01
        new_noise_spec = noise_spec * np.expand_dims(temp_factor, 1)
        new_denoised_spec = update_result(new_noise_spec, mix_spec)
        current_sim = calculate_similarity(
            denoised_spec=new_denoised_spec,
        )
        for i in range(LEN_BOUNDARY_INDEX-1):
            if current_sim[i] < min_sim[i]:
                min_sim[i] = current_sim[i]
                new_target_factor[BOUNDARY_INDEX[i]:BOUNDARY_INDEX[i+1]] = temp_factor[BOUNDARY_INDEX[i]]

    target_factor = new_target_factor

    new_noise_spec = noise_spec * np.expand_dims(target_factor, 1)
    spec_substraction_filter = 1 - (np.abs(new_noise_spec) / (np.abs(mix_spec) + 1e-20))
    spec_substraction_filter[spec_substraction_filter < 0] = 0
    denoised_spec = spec_substraction_filter * mix_spec
    
    condition = np.abs(denoised_spec) < np.abs(new_noise_spec) * freq_factors[:, np.newaxis]
    denoised_spec[condition] = 0

    return denoised_spec

def adaptive_spectral_substraction(noisy:np.ndarray,
                                   noise:np.ndarray,
                                   freq_factors:np.ndarray):
    len_input = len(noisy)
    if len_input < 1024:
        noisy = np.concatenate((noisy, np.zeros(1024-len(noisy))))
        noise = np.concatenate((noise, np.zeros(1024-len(noise))))

    [_, _, spec_noise] = scipy.signal.stft(
        noise,
        fs=FS,
        window='hann',
        nperseg=1024,
        noverlap=896,
        nfft=1024,
        scaling='psd',
        return_onesided=True,
        padded=True,
    )
    [_, _, spec_noisy] = scipy.signal.stft(
        noisy,
        fs=FS,
        window='hann',
        nperseg=1024,
        noverlap=896,
        nfft=1024,
        scaling='psd',
        return_onesided=True,
        padded=True,
    )

    denoised_spec = optimize_with_bs(
        mix_spec=spec_noisy,
        noise_spec=spec_noise,
        freq_factors=freq_factors
    )

    [_, denoised_result] = scipy.signal.istft(
        denoised_spec,
        fs=FS,
        window='hann',
        nperseg=1024,
        noverlap=896,
        nfft=1024,
        scaling='psd',
        input_onesided=True,
    )

    denoised_result = denoised_result[: len(noise)]

    return denoised_result

def corr_and_return(noise_chunk:np.ndarray, mix_chunk:np.ndarray):
    chunk_length = len(noise_chunk)
    corr_result = scipy.signal.correlate(noise_chunk, mix_chunk)
    idx = np.argmax(corr_result)

    if idx < chunk_length - 1:
        min_len = min(idx + 1, chunk_length)
        return noise_chunk[:min_len], mix_chunk[-min_len:], [idx + 1 - chunk_length, 0]
    elif idx >= chunk_length - 1:
        idx = 2 * chunk_length - 1 - idx
        min_len = min(idx, chunk_length)
        return noise_chunk[-min_len:], mix_chunk[idx - min_len : idx], [0, idx - chunk_length]

def chunk_spectral_subtraction(
        mix_chunk:np.ndarray,
        noise_chunk:np.ndarray,
        fs:int,
        freq_factors:np.ndarray=None
    ):
    noise_chunk, mix_chunk, [noise_index, mix_index] = corr_and_return(
        noise_chunk=noise_chunk, mix_chunk=mix_chunk
    )

    p_n = np.mean(noise_chunk * noise_chunk)
    p_m = np.mean(mix_chunk * mix_chunk)
    mix_chunk = mix_chunk * np.sqrt(p_n) / np.sqrt(p_m)

    clean_audio = adaptive_spectral_substraction(
            mix_chunk,
            noise_chunk,
            freq_factors
    )
    return clean_audio, mix_index, noise_index
