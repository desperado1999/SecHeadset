import os
import numpy as np
import soundfile as sf
import queue
import threading
import warnings
import time

from utils.parameters import *
from utils.funcs import generate_preamble, resample, decoding_from_spectrum
from src.stream_reader import Stream_Reader_sounddevice as Stream_Reader
from src.start_detector import start_detector
from utils.data_utils import *

# Test Cython
# from test_pyx.denoise_for_prototype import chunk_spectral_subtraction

# Not use Cython
from denoise_for_prototype import chunk_spectral_subtraction

class user_prototype:
    def __init__(self,
                 user_id:str,
                 partner_id:str,
                 phoneme_data_folder:str='./data/personal_dataset'
                 ):

        print('For denoising')
        self.lossy_factor_dict = np.load('./data/lossy_factor_telegram.npz')

        test_data = np.random.random(16000)
        [self.freqs, _, _] = scipy.signal.stft(
            test_data,
            fs=RATE,
            window='hann',
            nperseg=1024,
            noverlap=1,
            nfft=1024,
            scaling='psd',
            return_onesided=True,
            padded=False,
        )

        self.freq_factors = np.zeros(len(self.freqs))
        self.lossy_factor_freqs = self.lossy_factor_dict.files
        for i in range(len(self.lossy_factor_freqs)):
            closest_freqs = sorted(
                self.lossy_factor_freqs, key=lambda x_val: abs(int(x_val) - self.freqs[i])
            )[:2]
            self.freq_factors[i] = float((self.lossy_factor_dict[closest_freqs[0]] + self.lossy_factor_dict[closest_freqs[1]]) / 2)
        self.freq_factors.astype('float64')

        print('Init Parameter...')
        self._user_id = user_id
        self._partner_id = partner_id
        self._phoneme_data_folder = phoneme_data_folder

        # Set random seed
        self.random_seed = 1234

        # generate and read preamble
        if not os.path.exists('./data/preamble.wav'):
            self.preamble, preamble_fs = generate_preamble()
        else:
            self.preamble, preamble_fs = sf.read('./data/preamble.wav', samplerate=None)
            if preamble_fs != RATE:
                self.preamble, preamble_fs = generate_preamble()
        assert len(self.preamble) % CHUNK == 0, 'The length of preamble is not the integral multiple of CHUNK !!!'

        print('Read user&partner phoneme data...')
        # Read user&partner phoneme data and index
        self._user_vowel_data, self._user_vowel_index, self._user_consonant_data, self._user_consonant_index, self._user_phoneme_fs = self.read_phoneme_data(self._phoneme_data_folder, self._user_id)

        self._partner_vowel_data, self._partner_vowel_index, self._partner_consonant_data, self._partner_consonant_index, self._partner_phoneme_fs = self.read_phoneme_data(self._phoneme_data_folder, self._partner_id)

        print('Init queues...')
        # Init queues
        self.user_noise_queue = queue.Queue(maxsize=100)
        self.user_mix_queue = queue.Queue(maxsize=100)
        self.user_audio_queue = queue.Queue(maxsize=100)

        self.partner_noise_queue = queue.Queue(maxsize=100)
        self.partner_mix_queue = queue.Queue(maxsize=100)
        self.partner_audio_queue = queue.Queue(maxsize=100)

        print('Init streams...')
        self.in_stream = Stream_Reader(rate=RATE,
                            update_window_n_frames=UPDATE_WINDOW_N_FRAMES,
                            stream_type='in',
                            soundcard_keyword=SOUND_CARD_KEYWORD
                        )
        self.test_out_stream = Stream_Reader(rate=RATE,
                                        update_window_n_frames=UPDATE_WINDOW_N_FRAMES,
                                        out_data_queue=self.partner_audio_queue,
                                        stream_type='out',
                                        soundcard_keyword='USB'
                                        )
        self.out_stream = Stream_Reader(rate=RATE,
                                  update_window_n_frames=UPDATE_WINDOW_N_FRAMES,
                                  stream_type='out',
                                  out_data_queue=self.user_mix_queue,
                                  soundcard_keyword='USB'
                                )

        print('Init threads...')
        # Init threads

        # For user
        self.user_audio_file=  './data/103.wav'
        self.user_noise_thread = threading.Thread(
            target=self._noise_queue_thread,
            args=(self.user_noise_queue, 'user')
            )
        self.user_mix_thread = threading.Thread(target=self._mix_queue_thread)
        self.user_audio_thread = threading.Thread(
            target=self._audio_queue_thread,
            args=['user']
            )

        # For partner
        self.partner_noise_thread = threading.Thread(
            target=self._noise_queue_thread_with_file,
            args=(self.partner_noise_queue, './test.wav')
            )
        self.start_detection_residual_data = [None] * 1

        
        # Noise
        self.partner_start_detection_thread = threading.Thread(
            target=self._start_detection_thread,
            args=[self.in_stream, self.start_detection_residual_data],
        )
        self.partner_denoise_thread = threading.Thread(target=self._denoise_thread)

    def start_duplex(self):
        self.in_stream.start()
        self.partner_noise_thread.start()
        self.partner_start_detection_thread.start()
        self.partner_denoise_thread.start()

        # Transmit
        self.user_noise_thread.start()
        self.user_audio_thread.start()

        # Sleep for 2 seconds and put the stread_reader after noise and audio queue to make these two queue is ready for sending data for stream_reader
        time.sleep(2)
        self.user_mix_thread.start()
        self.out_stream.start()

        # Temp record
        audio = []
        for i in range(600):
            audio = np.concatenate((audio, self.partner_audio_queue.get()))
        sf.write('./temp.wav', audio, RATE)

    def start_transmit(self):
        self.user_noise_thread.start()
        self.user_audio_thread.start()

        time.sleep(2)
        self.stream_reader = Stream_Reader(rate=RATE,
                                           update_window_n_frames=UPDATE_WINDOW_N_FRAMES,
                                           stream_type='out',
                                           out_data_queue=self.user_mix_queue)

        self.user_mix_thread.start()

        self.stream_reader.start()

    def start_receive(self):
        # Init Stream_reader object
        self.in_stream.start()
        self.test_out_stream.start()

        self.partner_noise_thread.start()
        self.partner_denoise_thread.start()
        self.partner_start_detection_thread.start()

        input()
        os._exit(0)

    def start_receive_for_decoding(self):
        # Init Stream_reader object
        self.in_stream.start()
        self.partner_noise_thread.start()
        self.partner_start_detection_thread.start()

        # Decode modulated data
        while self.start_detection_residual_data[0] is None:
            continue
        ## Start decoding
        residual_data = self.start_detection_residual_data[0]
        residual_data = residual_data[- CHUNK * (len(residual_data) // CHUNK) : ]

        temp_index = 0
        while True:
            if abs(residual_data[temp_index]) > 0.2:
                break
            temp_index += 1

        residual_data = residual_data[temp_index:]

        start_point = 0
        get_decoded_data = 0

        # Read data
        expected_data = np.load('./data/temp_data.npy', allow_pickle=True)
        global_decode_data_counter = 1

        while start_point + CHUNK < len(residual_data):
            modulated_data, modulated_data1 = decoding_from_spectrum(np.abs(scipy.fft.fft(residual_data[start_point:start_point+CHUNK] * HAMM_WINDOW)))

            start_encoded_data = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
            start_encoded_data = start_encoded_data[:len(modulated_data)]
            if get_decoded_data == 0:

                if (modulated_data1 == start_encoded_data):
                    get_decoded_data = 1
                    print("Start Decoding!")

                    start_point += 2 * CHUNK
                else:
                    start_point += CHUNK_STEP
            else:
                if modulated_data == list(expected_data[global_decode_data_counter]):
                    print('Data Matched')
                else:
                    print(f'Data does not matched. Expected:{expected_data[global_decode_data_counter]}, but got {modulated_data}')

                global_decode_data_counter += 1
                if global_decode_data_counter == 11:
                    os._exit(1)

                start_point += 2 * CHUNK

        if len(residual_data) - start_point == 0:
            self.in_stream.in_queue.get()

        if get_decoded_data == 1:
            current_data = self.in_stream.in_queue.get()
            modulated_data = decoding_from_spectrum(np.abs(scipy.fft.fft(current_data * HAMM_WINDOW)))

            if modulated_data == list(expected_data[global_decode_data_counter]):
                print(1)
                print('Data Matched')
            else:
                print(1)
                print(f'Data does not matched. Expected:{expected_data[global_decode_data_counter]}, but got {modulated_data}')
            global_decode_data_counter += 1
            if global_decode_data_counter == 11:
                os._exit(1)

            self.in_stream.in_queue.get()

        else:
            print("Could not decode")
            os._exit(1)

        self.partner_denoise_thread.start()

        # Temp record
        audio = []
        for i in range(600):
            audio = np.concatenate((audio, self.partner_audio_queue.get()))
        sf.write('./temp.wav', audio, RATE)

    def _denoise_thread(self):
        noise_data_buffer = headset_data_buffer(capacity=2*RATE)
        noised_data_buffer = headset_data_buffer(capacity=2*RATE)
        chunk_size = 1024

        while True:
            noise_chunk = noise_data_buffer.get(data_len=chunk_size)
            while noise_chunk is None:
                noise_data_buffer.put(self.partner_noise_queue.get())
                noise_chunk = noise_data_buffer.get(data_len=chunk_size)

            noised_chunk = noised_data_buffer.get(data_len=chunk_size)
            while noised_chunk is None:
                noised_data_buffer.put(self.partner_mix_queue.get())
                noised_chunk = noised_data_buffer.get(data_len=chunk_size)

            audio_chunk, noised_index, noise_index = chunk_spectral_subtraction(noised_chunk, noise_chunk, self.freq_factors)

            noise_data_buffer.current_index += noise_index
            noised_data_buffer.current_index += noised_index

            self.partner_audio_queue.put(audio_chunk)

    def _mix_queue_thread(self):
        # This thread will only be used by 'user' and could not be used by 'partner'
        print('Mix queue thread start')

        # First put preamble to mix_queue
        idx  = 0
        while idx + CHUNK < len(self.preamble):
            self.user_mix_queue.put(self.preamble[idx : idx + CHUNK])
            idx += CHUNK

        while True:
            audio_chunk = self.user_audio_queue.get()
            noise_chunk = self.user_noise_queue.get()
            mix_chunk = 1.2 * (audio_chunk + noise_chunk)
            if np.max(np.abs(mix_chunk)) > 1:
                warnings.warn("The max amplitude of the mix audio is larger than 1 !!!") 
            self.user_mix_queue.put(mix_chunk)

    def _start_detection_thread(self, in_stream, residual_data):
        # Start detection
        detection_object = start_detector(RATE, CHUNK, in_stream.in_queue)

        recordings = list(detection_object.detect())

        print('Start Detected !!!')
        self.partner_mix_queue.put(recordings)
        while True:
            self.partner_mix_queue.put(in_stream.in_queue.get())

    def _audio_queue_thread(self, id):
        print('Audio queue thread start')

        if id == 'user':
            audio, fs = sf.read(self.user_audio_file, samplerate=None)
            if fs != RATE:
                audio = resample(audio, fs, RATE)
                fs = RATE

            idx = 0
            while idx + CHUNK < len(audio):
                self.user_audio_queue.put(audio[idx : idx + CHUNK])
                idx += CHUNK
        elif id == 'partner':
            raise NotImplementedError
        else:
            raise ValueError('The id must be \'user\' or \'partner\' !!!')

    def _noise_queue_thread_with_file(self, noise_queue, noise_file):
        noise_wav, _ = sf.read(noise_file)
        start_index = 0
        while start_index < len(noise_wav) - CHUNK:
            noise_queue.put(noise_wav[start_index:start_index+CHUNK])
            start_index += CHUNK

    def _noise_queue_thread(self, noise_queue, id):
        """
        Put sampled noise piece to noise queue, each item with a fixed length CHUNK
        """
        print('Noise queue thread start')

        # Set independent random number genertor
        rg_noise = np.random.default_rng(self.random_seed)
        noise_piece = np.array([0])

        if id == 'user':
            max_index = len(self._user_vowel_index)
            while True:
                if len(noise_piece) > CHUNK:
                    noise_queue.put(noise_piece[:CHUNK])
                    noise_piece = noise_piece[CHUNK:]
                else:
                    # Sample the noise
                    temp_index = rg_noise.integers(low=0, high=max_index, size=1)[0]
                    start_point = self._user_vowel_index[temp_index][0]
                    end_point = self._user_vowel_index[temp_index][1]
                    sampled_noise = self._user_vowel_data[start_point:end_point+1]

                    # Extend the sampled noise to noise_piece
                    noise_piece = np.concatenate((noise_piece, sampled_noise))
        elif id == 'partner':
            max_index = len(self._partner_vowel_index)
            while True:
                if len(noise_piece) > CHUNK:
                    noise_queue.put(noise_piece[:CHUNK])
                    noise_piece = noise_piece[CHUNK:]
                else:
                    # Sample the noise
                    temp_index = rg_noise.integers(low=0, high=max_index, size=1)[0]
                    start_point = self._partner_vowel_index[temp_index][0]
                    end_point = self._partner_vowel_index[temp_index][1]
                    sampled_noise = self._partner_vowel_data[start_point:end_point+1]

                    # Extend the sampled noise to noise_piece
                    noise_piece = np.concatenate((noise_piece, sampled_noise))
        else:
            raise ValueError('Wrong id type. The id must be \'user\' or \'partner\'.')

    def get_partner_info(self, partner_id, cipher_partner_uuid=0):
        self._partner_id = partner_id

        self._partner_vowel_data, self._partner_vowel_index, self._partner_consonant_data, self._partner_consonant_index, self._partner_phoneme_fs = self.read_phoneme_data(self._phoneme_data_folder, self._partner_id)

    @staticmethod
    def read_audio(audio_path:str):
        audio_wave, audio_wave_fs = sf.read(audio_path, samplerate=None)
        return audio_wave, audio_wave_fs

    @staticmethod
    def read_phoneme_data(data_folder, user_id):
        vowel_data_name = os.path.join(data_folder, user_id+'_vowel.wav')
        vowel_index_data_name = os.path.join(data_folder, user_id+'_vowel.csv')
        consonant_data_name = os.path.join(data_folder, user_id+'_consonant.wav')
        consonant_index_data_name = os.path.join(data_folder, user_id+'_consonant.csv')

        vowel_data, vowel_fs = sf.read(vowel_data_name, samplerate=None)
        # vowel_index = pd.read_csv(vowel_index_data_name, header=None)
        with open(vowel_index_data_name, 'r') as f:
            vowel_index = f.readlines()
        for i, item in enumerate(vowel_index):
            vowel_index[i] = list(map(int, item.strip().split(',')))

        consonant_data, consonant_fs = sf.read(consonant_data_name, samplerate=None)
        # consonant_index = pd.read_csv(consonant_index_data_name, header=None)
        with open(consonant_index_data_name, 'r') as f:
            consonant_index = f.readlines()
        for i, item in enumerate(consonant_index):
            consonant_index[i] = list(map(int, item.strip().split(',')))

        # For resample
        if vowel_fs != RATE:
            vowel_data = resample(vowel_data, vowel_fs, RATE)
            vowel_index = vowel_index * RATE / vowel_fs
            vowel_index = vowel_index.astype('int32')
            vowel_fs = RATE

        if consonant_fs != RATE:
            consonant_data = resample(consonant_data, consonant_fs, RATE)
            consonant_index = consonant_index * RATE / consonant_fs
            consonant_index = consonant_index.astype('int32')
            consonant_fs = RATE

        assert vowel_fs == consonant_fs, 'Error! The data sample rate do not match!'

        return vowel_data, vowel_index, consonant_data, consonant_index, vowel_fs

if __name__ == '__main__':
    test_class = user_prototype(user_id='103', partner_id='103')
    test_class.start_receive()
