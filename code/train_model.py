#!/usr/bin/env python
'''Model construction and training script'''

import argparse
import os
import pickle
import sys
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import keras as K

from tqdm import tqdm

from sklearn.model_selection import ShuffleSplit

import pumpp
import jams
import librosa


def process_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--max_samples', dest='max_samples', type=int,
                        default=128,
                        help='Maximum number of samples to draw per streamer')

    parser.add_argument('--patch-duration', dest='duration', type=float,
                        default=8.0,
                        help='Duration (in seconds) of training patches')

    parser.add_argument('--seed', dest='seed', type=int,
                        default='20170412',
                        help='Seed for the random number generator')

    parser.add_argument('--reference-path', dest='refs', type=str,
                        default=os.path.join(os.environ['HOME'],
                                             'eric_chords', 'references_v2'),
                        help='Path to reference annotations')

    parser.add_argument('--working', dest='working', type=str,
                        default=os.path.join(os.environ['HOME'],
                                             'working', 'chords'),
                        help='Path to working directory')

    parser.add_argument('--structured', dest='structured', action='store_true',
                        help='Enable structured training')

    parser.add_argument('--augmentation', dest='augmentation',
                        action='store_true',
                        help='Enable data augmentation')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Enable weighted sampling for training')

    parser.add_argument('--train-streamers', dest='train_streamers', type=int,
                        default=1024,
                        help='Number of active training streamers')

    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        default=32,
                        help='Size of training batches')

    parser.add_argument('--rate', dest='rate', type=int,
                        default=8,
                        help='Rate of pescador stream deactivation')

    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=100,
                        help='Maximum number of epochs to train for')

    parser.add_argument('--epoch-size', dest='epoch_size', type=int,
                        default=512,
                        help='Number of batches per epoch')

    parser.add_argument('--validation-size', dest='validation_size', type=int,
                        default=1024,
                        help='Number of batches per validation')

    parser.add_argument('--early-stopping', dest='early_stopping', type=int,
                        default=20,
                        help='Number of epochs without improvement to trigger early stopping')

    parser.add_argument('--reduce-lr', dest='reduce_lr', type=int,
                        default=10,
                        help='Number of epochs without improvement to trigger learning rate reduction')


    return parser.parse_args(args)


if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])
    print(params)
