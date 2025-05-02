import numpy as np
import sounddevice as sd
import queue
import sys
import scipy.signal

class Stream_Reader_sounddevice:
    """
    The Stread_Reader continuously reads data from a selected sound source using sounddevice
    """

    def __init__(self, rate, update_window_n_frames, stream_type, soundcard_keyword, out_data_queue=None):
        """
        Parameters:
            rate: audio sample rate, e.g., 16000, 48000 (Hz).
            out_data_queue: output data buffer
            update_window_n_frames: number of frames per seconds, e.g., 50. 
                This parameter also determine the length of each frame (rate/updata_window_n_frames). The length of each frame should not be too small, which could leads to data loss (The data reading speed is slower than the data sampling speed)
            stream_type: only input, only output, input + output
        """
        self.soundcard_keyword = soundcard_keyword
        self.in_queue = queue.Queue()
        self.out_queue = out_data_queue
        # self.rate = rate
        self.rate = 48000
        self.update_window_n_frames = update_window_n_frames
        self.frames_per_buffer = int(self.rate/self.update_window_n_frames)
        self.len_for_16k_data = int(self.frames_per_buffer/3)
        self.device = self.get_device_index()
        self.set_default_device(self.device, self.device)

        self.output_data_residual = []

        if stream_type == 'in':
            self.stream = sd.RawInputStream(
                samplerate=self.rate,
                blocksize=self.frames_per_buffer,
                channels=1,
                dtype=np.int16,
                callback=self.input_callback
            )

        elif stream_type == 'out':
            self.stream = sd.OutputStream(
                samplerate=self.rate,
                blocksize=self.frames_per_buffer,
                channels=1,
                dtype=np.float32,
                callback=self.output_callback
            )

        elif stream_type == 'inout':
            self.stream = sd.RawStream(
                samplerate=self.rate,
                blocksize=self.frames_per_buffer,
                channels=1,
                dtype=np.int16,
                callback=self.inout_callback
            )

        else:
            raise ValueError('Invalid stream type')
    
    def input_callback(self, in_data, frame_count, time_info, status):
        if status:
            print(status)

        # Input buffer

        ## For raspberry pi with 48kHz sample rate
        current_full_chunk = np.frombuffer(in_data, dtype=np.int16)/32768.0
        resampled_data = current_full_chunk[0::3]
        self.in_queue.put(resampled_data)

        ## For macos with 16kHz sample rate
        # current_full_chunk = np.frombuffer(in_data, dtype=np.int16)/32768.0
        # self.in_queue.put(current_full_chunk)

    def output_callback(self, out_data, frame_count, time_info, status):
        if status:
            print(status, file=sys.stderr)
            
        while len(self.output_data_residual) < self.len_for_16k_data:
            try:
                self.output_data_residual.extend(self.out_queue.get_nowait())
            except queue.Empty:
                continue
        data = self.output_data_residual[:self.len_for_16k_data]
        self.output_data_residual = self.output_data_residual[self.len_for_16k_data:]
        
        data = scipy.signal.resample(data, len(data)*3)


        if len(data) < len(out_data):
            out_data[:len(data)] = data
            out_data[len(data):] = b'\x00' * (len(out_data) - len(data))
            raise sd.CallbackStop
        else:
            out_data[:] = np.expand_dims(data, axis=1)     
    
    def inout_callback(self, in_data, out_data, frame_count, time_info, status):
        if status:
            print(status, file=sys.stderr)
        
        # Input buffer
        current_full_chunk = np.frombuffer(in_data, dtype=np.int16)/32768.0
        self.in_queue.put(current_full_chunk)

        # Output buffer
        if status.output_underflow:
            print('Output underflow: increase blocksize?', file=sys.stderr)
            raise sd.CallbackAbort
        try:
            data = self.out_queue.get_nowait()
        except queue.Empty:
            print('Buffer is empty: increase buffersize?', file=sys.stderr)
            raise sd.CallbackAbort
        if len(data) < len(out_data):
            out_data[:len(data)] = data
            out_data[len(data):] = b'\x00' * (len(out_data) - len(data))
            raise sd.CallbackStop
        else:
            out_data[:] = np.expand_dims(data, axis=1)

    def non_blocking_callback(self,
                              in_data,
                              out_data,
                              frame_count,
                              time_info,
                              status
                            ):
        if status:
            print(status, file=sys.stderr)
        
        # Input buffer
        current_full_chunk = np.frombuffer(in_data, dtype=np.int16)/32768.0
        self.in_queue.put(current_full_chunk)

        # Output buffer
        if status.output_underflow:
            print('Output underflow: increase blocksize?', file=sys.stderr)
            raise sd.CallbackAbort
        try:
            data = self.out_queue.get_nowait()
        except queue.Empty:
            print('Buffer is empty: increase buffersize?', file=sys.stderr)
            raise sd.CallbackAbort
        if len(data) < len(out_data):
            out_data[:len(data)] = data
            out_data[len(data):] = b'\x00' * (len(out_data) - len(data))
            raise sd.CallbackStop
        else:
            out_data[:] = np.expand_dims(data, axis=1)
    
    def start(self):
        self.stream.start()
        
    def stop(self):
        print("Sending stream termination command...")
        self.stream.stop()
        self.stream.close()

    def get_device_index(self):
        devices = sd.query_devices()
        for i in range(len(devices)):
            if self.soundcard_keyword in devices[i]['name']:
                return i
        raise Exception('Do not find target soundcard')

    @staticmethod
    def set_default_device(input_device_index, output_device_index):
        sd.default.device = [input_device_index, output_device_index]