import os
import soundfile as sf
import scipy.signal
import numpy as np

def corr_and_return(noise_chunk, mix_chunk):
    chunk_length = len(noise_chunk)

    corr_result = scipy.signal.correlate(noise_chunk, mix_chunk)
    idx = np.argmax(corr_result)
    
    if idx < len(noise_chunk) - 1:
        min_len = min(idx + 1, len(mix_chunk))
        mix_chunk = mix_chunk[-min_len:]
        noise_chunk = noise_chunk[: min_len]

        return noise_chunk, mix_chunk, [idx + 1 - chunk_length, 0]

    elif idx > len(noise_chunk) - 1:
        idx = len(noise_chunk) + len(mix_chunk) - 1 - idx
        min_len = min(idx, len(noise_chunk))
        noise_chunk = noise_chunk[-min_len:]
        mix_chunk = mix_chunk[idx-min_len:idx]

        return noise_chunk, mix_chunk, [0, idx - chunk_length]
    else:
        length = min(len(mix_chunk), len(noise_chunk))
        noise_chunk = noise_chunk[-length:]
        mix_chunk = mix_chunk[-length:]  
        
        return noise_chunk, mix_chunk, [0,0]