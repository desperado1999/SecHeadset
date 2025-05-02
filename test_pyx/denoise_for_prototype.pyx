cimport numpy as np
import numpy as np
from scipy.signal import stft, istft, correlate

cdef double MAX_AMP = 0.9
cdef int FS = 16000
cdef int LEN_FREQS = 513
cdef int LEN_BOUNDARY_INDEX = 12
FREQS = np.linspace(0,8000,513)
BOUNDARY = [20, 160, 394, 670, 1000, 1420, 1900, 2450, 3120, 4000, 5100, 6600]
BOUNDARY_INDEX = np.array([1, 10, 25, 43, 64, 91, 122, 157, 200, 256, 326, 422])
FACTOR_RANGE = np.arange(0.2, 1.2, 0.1)

def update_result(np.ndarray[np.complex128_t, ndim=2] noise_spec,
                  np.ndarray[np.complex128_t, ndim=2] mix_spec):
    
    cdef np.ndarray[double, ndim=2] spec_substraction_filter = 1 - (np.abs(noise_spec) / (np.abs(mix_spec) + 1e-30))
    cdef np.ndarray[np.complex128_t, ndim=2] denoised_spec = spec_substraction_filter * mix_spec
    return denoised_spec

def calculate_similarity(np.ndarray[np.complex128_t, ndim=2] denoised_spec):
    cdef np.ndarray[double, ndim=1] sim = np.zeros(LEN_BOUNDARY_INDEX-1)

    cdef int i
    for i in range(LEN_BOUNDARY_INDEX-1):
        sim[i] = np.sum(
            np.abs(denoised_spec[BOUNDARY_INDEX[i] : BOUNDARY_INDEX[i + 1], :])
        )   
    return sim

def optimize_with_bs(np.ndarray[np.complex128_t, ndim=2] mix_spec,
                     np.ndarray[np.complex128_t, ndim=2] noise_spec,
                     np.ndarray[double, ndim=1] freq_factors):
    cdef np.ndarray[double, ndim=1] target_factor = np.ones(LEN_FREQS) * 0.2
    cdef np.ndarray[double, ndim=1] min_sim = 1e6 * np.ones(LEN_BOUNDARY_INDEX - 1)
    cdef np.ndarray[np.complex128_t, ndim=2] new_noise_spec, new_denoised_spec
    cdef np.ndarray[double, ndim=1] current_sim
    cdef int i
    cdef double temp_factor_1
    cdef np.ndarray[double, ndim=1] temp_factor

    for temp_factor_1 in FACTOR_RANGE:
        factor = np.ones(LEN_FREQS) * temp_factor_1
        new_noise_spec = noise_spec * np.expand_dims(factor, 1)
        new_denoised_spec = update_result(new_noise_spec, mix_spec)
        current_sim = calculate_similarity(new_denoised_spec)
        for i in range(LEN_BOUNDARY_INDEX-1):
            if current_sim[i] < min_sim[i]:
                min_sim[i] = current_sim[i]
                target_factor[BOUNDARY_INDEX[i]:BOUNDARY_INDEX[i+1]] = temp_factor_1

    new_target_factor = np.copy(target_factor)

    for i in range(1, 21):
        temp_factor = target_factor - 0.1 + i * 0.01
        new_noise_spec = noise_spec * np.expand_dims(temp_factor, 1)
        new_denoised_spec = update_result(new_noise_spec, mix_spec)
        current_sim = calculate_similarity(new_denoised_spec)
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

def adaptive_spectral_substraction(np.ndarray[np.float64_t, ndim=1] noisy,
                                   np.ndarray[np.float64_t, ndim=1] noise,
                                   np.ndarray[np.float64_t, ndim=1] freq_factors
                                   ):
    cdef int len_input = len(noisy)
    cdef np.ndarray[np.float64_t, ndim=1] padded_noisy
    cdef np.ndarray[np.float64_t, ndim=1] padded_noise

    if len_input < 1024:
        padded_noisy = np.concatenate((noisy, np.zeros(1024-len(noisy))))
        padded_noise = np.concatenate((noise, np.zeros(1024-len(noise))))
    else:
        padded_noise = noise
        padded_noisy = noisy

    [_, _, spec_noise] = stft(
        padded_noise,
        FS,
        'hann',
        1024,
        896,
        1024,
        False,
        True,
        'zeros',
        'True', 
        -1,
        'psd',
    )
    [_, _, spec_noisy] = stft(
        padded_noisy,
        FS,
        'hann',
        1024,
        896,
        1024,
        False,
        True,
        'zeros',
        'True', 
        -1,
        'psd',
    )

    denoised_spec = optimize_with_bs(
        mix_spec=spec_noisy,
        noise_spec=spec_noise,
        freq_factors=freq_factors
    )

    [_, denoised_result] = istft(
        denoised_spec,
        FS,
        'hann',
        1024,
        896,
        1024,
        True,
        True,
        -1, 
        -2,
        'psd',
    )

    denoised_result = denoised_result[: len(noise)]

    return denoised_result

def corr_and_return(np.ndarray[np.float64_t, ndim=1] noise_chunk,
                    np.ndarray[np.float64_t, ndim=1] mix_chunk
                    ):
    # len(noise_chunk) must equal to len(mix_chunk)
    cdef int chunk_length = len(noise_chunk)
    cdef np.ndarray[np.float64_t, ndim=1] corr_result = correlate(noise_chunk, mix_chunk)
    cdef int idx = np.argmax(corr_result)
    cdef int min_len

    if idx < chunk_length - 1:
        min_len = min(idx + 1, chunk_length)
        return noise_chunk[:min_len], mix_chunk[-min_len:], [idx + 1 - chunk_length, 0]
    elif idx >= chunk_length - 1:
        idx = 2 * chunk_length - 1 - idx
        min_len = min(idx, chunk_length)
        return noise_chunk[-min_len:], mix_chunk[idx - min_len : idx], [0, idx - chunk_length]

def chunk_spectral_subtraction(
        np.ndarray[np.float64_t, ndim=1] mix_chunk,
        np.ndarray[np.float64_t, ndim=1] noise_chunk,
        np.ndarray[np.float64_t, ndim=1] freq_factors
    ):
    cdef int noise_index, mix_index
    cdef double p_n, p_m
    cdef np.ndarray[np.float64_t, ndim=1] clean_audio
    cdef np.ndarray[np.float64_t, ndim=1] new_noise_chunk, new_mix_chunk


    noise_chunk, mix_chunk, [noise_index, mix_index] = corr_and_return(
        noise_chunk, mix_chunk
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
