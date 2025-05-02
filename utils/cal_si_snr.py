import numpy as np

smallVal = np.finfo("float").eps  # To avoid divide by zero

def si_snr(ref, noisy):

    assert len(ref) == len(noisy), 'The lengths of the inputs do not match!!!'

    dot = np.sum(ref * noisy, keepdims=True)
    s_ref_energy = np.sum(ref ** 2, keepdims=True) + smallVal
    proj = dot * ref / s_ref_energy
    e_noise = noisy - proj
    si_snr_beforelog = np.sum(proj**2) / (np.sum(e_noise ** 2) + smallVal)
    return 10 * np.log10(si_snr_beforelog + smallVal)