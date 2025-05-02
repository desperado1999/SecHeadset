import numpy as np

class headset_data_buffer:
    def __init__(self, capacity):
        # self.capacity = 5 * RATE # 5 seconds 
        self.capacity = capacity
        self.data  = np.zeros(self.capacity, dtype=np.float64)
        self.current_index = 0
        self.size = 0
    
    def put(self, value_list):
        assert len(value_list) > 0, "The input must be a not-none list!!!"

        if self.size + len(value_list) > self.capacity:
            print(self.size)
            print(len(value_list))
            print(self.capacity)
            raise ValueError('Data buffer overflow!!!')
        else:
            self.data[self.size : self.size + len(value_list)] = value_list
            self.size += len(value_list)

    def get(self, data_len, step_len=None):
        """
        Inputs:
            data_len: the length of the required output data
            step_len: the length of the step for the current index
        """
        if step_len == None:
            step_len = data_len

        if self.size - self.current_index < data_len:
            # print("Not enough data in the buffer")
            return None
        else:
            self._remove_useless_data()
            output_data = self.data[self.current_index : self.current_index + data_len]
            self.current_index += step_len
            return output_data
    
    def _remove_useless_data(self):
        # Remove data before current_index
        self.data = np.roll(self.data, -self.current_index)
        self.size = self.size - self.current_index
        self.current_index = 0

    def __call__(self):
        return self.data[:self.size]
