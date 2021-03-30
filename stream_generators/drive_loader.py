import itertools
import logging
import os
import pickle
import random
from abc import abstractmethod
from os import path

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


class ClipLoader(data.Dataset):
    def __init__(self,
                 signals_dataset_path,
                 signals_input=[],
                 window_length=10,
                 history_stride=1,
                 show_errors=False,
                 print_description=True,
                 flip_prob=0.5,
                 ignore_sensors_for_normalization=None,
                 split_type='train',
                 samples_interval=0.1,
                 ):
        """

        Args:
            signals_input (list(str)): signals to use for input
            signals_dataset_path (str): path to directory of npz signals files
            show_errors (bool): whether to log errors from loading videos and signals
            flip_prob (float): probability of flipping the video horizontally. This also followed by inverting the lean
        """
        super(ClipLoader, self).__init__()

        # loader params
        self.show_errors = show_errors
        self.sample_size = None
        self.flip_prob = flip_prob
        self.samples_interval = samples_interval
        self.split_type = split_type

        # signals io
        self.signals_dataset_path = signals_dataset_path

        # input signals
        self.signals_input = signals_input
        self.window_length = window_length
        self.history_stride = history_stride

        # prepare data
        self.ignore_sensors_for_normalization = ignore_sensors_for_normalization
        self.video_names = self._get_video_list()
        self.sensors_data = self._load_sensors_data()  # dictionary rider_name |--> sensors data ()
        self.idx_2_video = [k for k, v in self.sensors_data.items()]
        self.sensors_data, self.unnormalized_sensor_data, self.sensor_means, self.sensor_stds, self.sensor_maxs = self.normalize_data(
            self.sensors_data)
        self.lengths = self.split_idx()
        # self.filter_missing_files()

        if print_description:
            print(repr(self))

        self.valid_idx = set(range(len(self)))

    def set_flip_prob(self, flip_prob):
        self.flip_prob = flip_prob

    def output_sensor_idx(self, sensor_name):
        return self.signals_output.index(sensor_name)

    def sample(self, size):
        if size > len(self):
            return

        self.sample_size = size

    @property
    def unnormalized_output_values(self):
        """
        Retrieves all observations from all the data stored in the loader for the target variables
        Returns:

        """
        data = [v for k, v in self.unnormalized_sensor_data.items()]
        data = pd.concat(data)
        return data[self.signals_output]

    @property
    def normalized_output_values(self):
        """
        Retrieves all observations from all the data stored in the loader for the target variables
        """
        data = [v for k, v in self.sensors_data.items()]
        data = pd.concat(data)
        return data[self.signals_output]

    def apply_normalization(self, data, mean_value, std_value):
        for col in data.columns:
            data[col] = (data[col] - mean_value[col]) / std_value[col]

        return data

    def renormalize_data(self, mean_value, std_value):
        data = self.unnormalized_sensor_data

        if type(data) == list:
            normalized_data = [(d - mean_value) / (std_value + 1e-7) for d in data]
        elif type(data) == dict:
            normalized_data = {k: (d - mean_value) / (std_value + 1e-7) for k, d in data.items()}

        self.sensor_means = mean_value
        self.sensor_stds = std_value
        self.sensors_data = normalized_data

    def normalize_data(self, data):
        """
        Normalizes the data using z-score
        Args:
            data: list or dictionary of data-frames containing the signals data

        Returns: list of data frames, with normalized values
        """
        if type(data) == dict:
            data_to_normalize = [v for k, v in data.items()]

        conc_data = pd.concat(data_to_normalize)
        max_value = conc_data.max()
        mean_value = conc_data.mean()
        conc_data -= mean_value
        std_value = conc_data.std()
        # std_value = conc_data.std()
        if self.ignore_sensors_for_normalization is not None:
            for sensor_name in self.ignore_sensors_for_normalization:
                mean_value[sensor_name] = 0
                std_value[sensor_name] = 1

        mean_value['time'] = 0
        std_value['time'] = 1

        if type(data) == list:
            normalized_data = [(d - mean_value) / (std_value + 1e-7) for d in data]
        elif type(data) == dict:
            normalized_data = {k: (d - mean_value) / (std_value + 1e-7) for k, d in data.items()}

        return normalized_data, data, mean_value, std_value, max_value

    def get_denormalization_for_sensor(self, sensor_name):
        def inner(sensor_data):
            return sensor_data * self.sensor_stds[sensor_name] + self.sensor_means[sensor_name]

        return inner

    def unnormalize_targets(self, sensors_data, sensors_names):
        return [sensor_data * self.sensor_stds[sensor_name] + self.sensor_means[sensor_name]
                for sensor_data, sensor_name in zip(sensors_data, sensors_names)]

    def mins(self):
        return self.unnormalized_output_values.min()

    def maxs(self):
        return self.unnormalized_output_values.max()

    def get_min_value_for_sensor(self, sensor_name):
        return min([data[sensor_name].min() for data in self.sensors_data])

    def get_max_value_for_sensor(self, sensor_name):
        return max([data[sensor_name].max() for data in self.sensors_data])

    def get_loader_shape(self):
        """
        Returns: the shape of a video sample
        """
        return self[0].shape

    def __repr__(self):
        return f"""
===== ClipLoader =====
signals_input = {self.signals_input}
signals_dataset_path = {self.signals_dataset_path}
number of files = {len(self.video_names)}
loader length = {len(self)}
show_errors = {self.show_errors}
"""

    @abstractmethod
    def _load_sensors_data(self):
        """
        loads the sensors data for all riders
        :return dictionary video -> pandas dataframe (one column for each sensor)
        """
        assert os.path.exists(self.signals_dataset_path), "VideoIter:: failed to locate: `{}'".format(
            self.video_dataset_path)
        return None, None, None

    @abstractmethod
    def _get_video_list(self):
        """
        Scans the directory pointed by video_dataset_path and retrieves all paths to riders videos
        :return tuple (video paths, video names)
        """
        assert os.path.exists(self.signals_dataset_path), "VideoIter:: failed to locate: `{}'".format(
            self.signals_dataset_path)
        return None, None

    def split_idx(self):
        lengths = []
        for k, data_df in self.sensors_data.items():
            length = len(data_df)
            num_samples = length - self.window_length + 1  # amount of chunks series in one file
            lengths.append(num_samples)

        return np.cumsum(lengths)

    def __len__(self):
        """
        Retrieves the length of the dataset.
        """
        return self.lengths[-1] if len(self.lengths) != 0 else 0

    def __getitem__(self, index):
        """
        Method to access the i'th sample of the dataset
        Args:
            index: the sample index

        Returns: the i'th sample of this dataset
        """
        succ = False
        while not succ:
            try:
                data = self.getitem_from_raw_video(index)
                succ = True
            except Exception as e:
                if index in self.valid_idx:
                    self.valid_idx.remove(index)

                index = random.sample(self.valid_idx, 1)[0]

                if self.show_errors:
                    logging.warning(e)

        return data

    def get_clip_properties(self, idx):
        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        video_name = self.video_names[video_idx]
        return video_idx, clip_idx, video_name

    def prepare_video_clip(self, idx):
        """
        Retrieves a video sample and takes care of all preparation work of it
        Args:
            idx: the index of the clip in the current dataset
        Returns:
            video - a video clip ready as input to a network
            video_idx - index of the corresponding video in the dataset
            clip_idx - index of the current clip in the corresponding video
            video_name - the name of the video the retrieved video clip is taken from

        """
        video, _, _, _ = self.video_clips.get_clip(idx)

        in_clip_frames = list(range(0, self.total_clip_length_in_frames, self.frame_stride))
        video = video[in_clip_frames]
        transformed_video = None
        if self.video_transform is not None:
            transformed_video = self.video_transform(video)

        return transformed_video, video

    def load_signals(self, idx):
        # video_idx, clip_idx, video_name = self.get_clip_properties(idx)
        video_idx = np.searchsorted(self.lengths, idx)
        video_name = self.idx_2_video[video_idx]
        data_df = self.unnormalized_sensor_data[video_name]
        data_df = data_df[self.signals_input]
        if video_idx >= 1:
            idx -= self.lengths[video_idx - 1]

        if video_name not in self.sensors_data:
            raise Exception(f"No signals data for {video_name}")

        return data_df.copy(), idx

    def to_chunks(self, data_df, idx):
        end = idx + self.window_length * self.history_stride
        if end >= len(data_df):
            raise IndexError("Out of bounds of dataframe")
        data_df = data_df[self.signals_input].values[idx:  idx + self.window_length * self.history_stride, :]
        # steps = list(range(idx, idx + self.num_chunks * self.samples_per_chunk + 1, self.samples_per_chunk))
        # chunks = [data_df[i_start: i_end, :] for i_start, i_end in zip(steps[:-1], steps[1:])]
        # return torch.tensor(np.array(chunks)).transpose(1, 2)
        return torch.tensor(data_df).transpose(0, 1).float()

    def load_video(self, idx):
        if not self.return_video:
            return np.array([1]), np.array([1])  # return a dummy output if not expected to use the video

        transformed_video, video = self.prepare_video_clip(idx)

        if self.input_type == 'img':
            video = video.squeeze(1)

        return transformed_video, video

    def getitem_from_raw_video(self, idx):
        """
        Retrieves all relevant information for iterating over the dataset.
        Returns: corresponding training sample
        """

        # Signals data loading
        data_df, idx = self.load_signals(idx)

        # Augment by flipping horizontally
        if self.split_type != 'validation':
            if 'steering_angle' in data_df.columns:
                steering_name = 'steering_angle'
            elif 'steering' in data_df.columns:
                steering_name = 'steering'
            else:
                steering_name = None

            if random.random() < self.flip_prob and steering_name is not None:
                data_df = data_df.copy()
                data_df[steering_name] *= -1

        data_df = self.apply_normalization(data_df, self.sensor_means, self.sensor_stds)
        signals = self.to_chunks(data_df=data_df, idx=idx)

        return signals

    def filter_missing_files(self):
        """
        Filters riders that don't have either corresponding video or signals information
        """
        data_in_videos_and_sensors = set(self.video_names).intersection(list(self.sensors_data.keys()))
        all_available_data = set(self.video_names).union(list(self.sensors_data.keys()))
        exist_idx = [idx for idx, name in enumerate(self.video_names) if name in data_in_videos_and_sensors]
        self.video_names = [self.video_names[idx] for idx in exist_idx]
        self.video_paths = [self.video_paths[idx] for idx in exist_idx]

        if self.show_errors:
            logging.warning(
                f"Missing video/signals data for {len(all_available_data - data_in_videos_and_sensors)} files:")
            for name in all_available_data - data_in_videos_and_sensors:
                logging.warning(f'        {name}')