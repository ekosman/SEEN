import os
from os import path
import pandas as pd
import numpy as np
import torch

from tqdm import tqdm
from scipy.interpolate import interp1d

from stream_generators.drive_loader import ClipLoader


class CommaLoader(ClipLoader):
    def __init__(self,
                 **kwargs,
                 ):
        # call the super class with the new arguments arrangement
        super(CommaLoader, self).__init__(**kwargs)

    def _get_video_list(self):
        super(CommaLoader, self)._get_video_list()
        print("Creating video list...")
        video_names = []
        for root, dirs, files in os.walk(self.signals_dataset_path):
            for file in files:
                if file.endswith(r"preview.png"):
                    file_path = os.path.join(root, file)
                    video_names.append(file_path.split(r'preview.png')[0])

        return video_names

    def _load_sensors_data(self):
        super(CommaLoader, self)._load_sensors_data()
        print("Loading data...")
        data = dict()
        can_signals = ['speed', 'steering_angle', 'wheel_speed']
        imu_signals = ['accelerometer', 'gyro', 'gyro_bias', 'gyro_uncalibrated']
        all_signals = can_signals + imu_signals
        signals = []

        i = 1
        for video_name in tqdm(self.video_names):
            signals = []
            interpolators = dict()
            start_time = - float('inf')
            end_time = float('inf')

            # Load times and signals
            for signal in all_signals:
                if signal in can_signals:
                    signal_dir = path.join(self.signals_dataset_path, video_name, 'processed_log', 'CAN', signal)
                elif signal in imu_signals:
                    signal_dir = path.join(self.signals_dataset_path, video_name, 'processed_log', 'IMU', signal)
                else:
                    raise Exception(f"Unknown signal {signal}")

                time = np.load(path.join(signal_dir, 't')).flatten()
                start_time = max(start_time, time.min())
                end_time = min(end_time, time.max())

                values = np.load(path.join(signal_dir, 'value'))

                if len(values.shape) == 1:
                    interpolators[signal] = interp1d(time, values)
                    signals.append(signal)
                else:
                    for dim_i in range(values.shape[1]):
                        signal_name = f"{signal}_{dim_i}" if values.shape[1] > 1 else signal
                        signals.append(signal_name)
                        interpolators[signal_name] = interp1d(time, values[:, dim_i])

            df = pd.DataFrame(columns=signals + ['time'])
            # quantize all signals
            df['time'] = np.arange(start_time, end_time, self.samples_interval)
            for signal in signals:
                df[signal] = interpolators[signal](df['time'])

            df['time'] -= start_time
            data[video_name] = df

            # if i == 5:
            #     break

            i += 1

        self.all_signals = signals

        return data


if __name__ == '__main__':
    data = CommaLoader(signals_dataset_path=r'C:\comma\comma2k19\all_data',
                       samples_interval=0.005, signals_input=['steering_angle', 'speed'])
    data[1000]
    loader = torch.utils.data.DataLoader(data,
                                         batch_size=5,
                                         shuffle=True,
                                         num_workers=4,
                                         pin_memory=True)

    print(data.get_loader_shape())

    for batch in tqdm(loader):
        pass
