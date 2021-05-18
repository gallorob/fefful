import configparser
import os

import torch as th
from torch import sigmoid

from minecraft_pb2 import *


class MCSettings:
    def __init__(self,
                 filename: str):
        config = configparser.ConfigParser()
        config.read(filename)
        self.start_coordinates = [
            config['START POSITION'].getint('x'),
            config['START POSITION'].getint('y'),
            config['START POSITION'].getint('z')
        ]

        self.artifact_dimensions = [
            config['ARTIFACT'].getint('width'),
            config['ARTIFACT'].getint('height'),
            config['ARTIFACT'].getint('depth')
        ]
        self.artifact_spacing = config['ARTIFACT'].getint('spacing')
        self.evaluatable_artifacts = config['ARTIFACT'].getint('evaluatable_artifacts')

        self.admissible_rotations = config['ADMISSIBILES'].get('rotations').replace(' ', '').upper().split(',')
        self.admissible_blocks = config['ADMISSIBILES'].get('blocks').replace(' ', '').upper().split(',')

        self.buffer_capacity = config['ESTIMATOR'].getint('buffer_capacity')
        self.batch_size = config['ESTIMATOR'].getint('batch_size')
        self.conv_channels = config['ESTIMATOR'].getint('conv_channels')
        self.train_epochs = config['ESTIMATOR'].getint('train_epochs')
        self.train_interval = config['ESTIMATOR'].getint('train_interval')
        self.test_threshold = config['ESTIMATOR'].getfloat('test_threshold')

        self.nets_folder = os.path.join(os.path.dirname(__file__), config['ESTIMATOR'].get('nets_folder'))
        os.makedirs(self.nets_folder, exist_ok=True)

    @property
    def x0(self):
        return self.start_coordinates[0]

    @property
    def y0(self):
        return self.start_coordinates[1]

    @property
    def z0(self):
        return self.start_coordinates[2]

    @property
    def artifact_width(self):
        return self.artifact_dimensions[0]

    @property
    def artifact_depth(self):
        return self.artifact_dimensions[2]

    @property
    def artifact_height(self):
        return self.artifact_dimensions[1]

    def val_to_enum(self,
                    val: float,
                    block: bool = True):
        l = self.admissible_blocks if block else self.admissible_rotations
        e = BlockType if block else Orientation
        return e.Value(l[int(sigmoid(th.as_tensor(val)).numpy() * len(l)) - 1])
