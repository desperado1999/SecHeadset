import numpy as np
import scipy.signal
import scipy

from utils.funcs import corr_2d
from utils.parameters import *
from utils.data_utils import *

class start_detector:
    """
    Detect transmission start with self correlation
    """

    def __init__(self, rate, chunk_size, input_queue):
        """
        Inputs:
            rate: audio sample rate / Hz
            chunk_size: audio reader chunk size
            input_queue: Input data queue
        """

        self.rate = rate
        self.chunk_size = chunk_size
        self.input_queue = input_queue

    def detect(self):
       while True: 
            print('Start Detection')
            data_buffer = headset_data_buffer(capacity=5*RATE)
            restart_flag = 0
            while True:
                current_chunk = self.input_queue.get()
                if self.coarse_start_detection(current_chunk):
                    print("Coarse start flag detected!")
                    break

            for _ in range(int(0.7*RATE/CHUNK)):
                data_buffer.put(self.input_queue.get())
            ## Calculation corr
            temp_input_data = data_buffer()
            temp_input_data = temp_input_data / max(abs(temp_input_data))
            normalized_start_pattern = START_PATTERN / max(abs(START_PATTERN))

            corr_result = scipy.signal.correlate(temp_input_data,
                                                 normalized_start_pattern,
                                                 method='direct',
                                                 mode='full')
            peak_loc = self.peak_detection(corr_result)
            if peak_loc:
                print("Precise start detected.")
                return temp_input_data[peak_loc:]
            else:
                print("Start detecton failed. Restart start detection.")
                continue
            
    @staticmethod
    def peak_detection(corr_buffer):
        [peaks, properties] = scipy.signal.find_peaks(
            corr_buffer, height=0.08, distance=10
            )
        if len(peaks) ==  1:
            return peaks[0]
        elif len(peaks) > 1:
            peak_heights = properties['peak_heights']
            target_index = np.argmax(peak_heights)
            return peaks[target_index]
        else:
            return None
            
    @staticmethod
    def coarse_start_detection(data_chunk):
        assert len(data_chunk) == CHUNK, "The length of input data is not CHUNK!"

        min_freq = TONE_FREQ - 500
        target_freq = TONE_FREQ
        max_freq = TONE_FREQ + 500
        threshold = 20
        
        min_freq_index = np.argmin(np.abs(FFT_CHUNK_FREQ - min_freq))
        max_freq_index = np.argmin(np.abs(FFT_CHUNK_FREQ - max_freq))

        target_freq_index = np.argmin(np.abs(FFT_CHUNK_FREQ - target_freq))
        
        fft_result = scipy.fft.fft(data_chunk)
        abs_fft_result = np.abs(fft_result)
        
        target_freq_value = np.mean(
            abs_fft_result[target_freq_index - 1 : target_freq_index + 1]
        )

        residual_value = (
            np.sum(abs_fft_result[min_freq_index:target_freq_index - 1]) +
            np.sum(abs_fft_result[target_freq_index + 1 : max_freq_index])
            ) / (max_freq_index - min_freq_index)

        return target_freq_value > threshold * residual_value


