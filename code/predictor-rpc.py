#!/bin/env python

import argparse
import pickle
import os
import sys
import keras as K
from tqdm import tqdm
import pandas as pd
import numpy as np

import jams


def mkout(output_dir, filename):

    subidx = os.path.basename(os.path.dirname(filename))

    subdir = os.path.join(output_dir, subidx)

    jams.util.smkdirs(subdir)

    baseidx = os.path.splitext(os.path.basename(filename))[0]

    return os.path.join(subdir, os.path.extsep.join([baseidx, 'npz']))


def predict_example(model, pump, audio_file, output_dir):

    data = pump.transform(filename=audio_file)

    p_tag, p_pc, p_root, p_bass = model.predict(data['cqt/mag'])

    # Save p_root, p_pc to disk
    outfile = mkout(output_dir, audio_file)
    np.savez(outfile, p_pc=p_pc, p_root=p_root)


def test_fold(pump, model_dir, fold_number, index_file, output_dir):

    model = K.models.load_model(os.path.join(model_dir,
                                             'fold{:02d}_weights.pkl'.format(fold_number)),
                                custom_objects=dict(K=K),
                                compile=False)

    test_files = pd.read_csv(index_file, header=None, squeeze=True)

    for filename in tqdm(test_files, desc='Predicting'):
        predict_example(model, pump, filename, output_dir)


def evalutron(working, model_dir, fold, index_file, output_dir):

    pump = pickle.load(open(os.path.join(working, 'pump.pkl'), 'rb'))

    test_fold(pump, model_dir, fold, index_file, output_dir)


def parse_args(args):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--working', dest='working', type=str,
                        default=os.path.join(os.environ['HOME'],
                                             'working', 'chords'),
                        help='Path to working directory')

    parser.add_argument('--fold-num', dest='fold_num', type=int,
                        default=0,
                        help='Index of model fold to use')

    parser.add_argument(dest='model_dir', type=str,
                        help='Path to model directory')

    parser.add_argument(dest='index_file', type=str,
                        help='Path to file index')

    parser.add_argument(dest='output_dir', type=str,
                        help='Path to save outputs')

    return parser.parse_args(args)


if __name__ == '__main__':
    params = parse_args(sys.argv[1:])
    evalutron(params.working,
              params.model_dir,
              params.fold_num,
              params.index_file,
              params.output_dir)
