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

# Load the model
#   model = K.models.load_model('foldXX_weights.pkl',
#                               custom_objects=dict(K=K),
#                               compile=False)
#
# Load the test points
#   test_ids = pd.read_csv('testXX.csv', header=None, squeeze=True)
#
# Iterate over test points for prediction
#
#   for fn in test_ids:
#       data = dict(np.load('../pump/{}.npz'))
#       preds = model.predict(data['cqt/mag'])
#       ann = P['chord_tag'].inverse(preds[1][0])
#           duration = ann.data[-1].time + ann.data[-1].duration
#       stash in a jams object J
#       save jams out


def predict_example(model, pump, file_id, working, model_dir):

    data = dict(np.load(os.path.join(working, 'pump', '{}.npz'.format(file_id))))
    preds = model.predict(data['cqt/mag'])
    ann = pump['chord_tag'].inverse(preds[1][0])
    J = jams.JAMS()
    J.file_metadata.identifiers['track_id'] = file_id
    J.file_metadata.duration = ann.data[-1].time + ann.data[-1].duration
    J.annotations.append(ann)
    J.save(os.path.join(model_dir, 'predictions', '{}.jams'.format(file_id)))


def test_fold(pump, working, model_dir, fold_number):

    model = K.models.load_model(os.path.join(model_dir,
                                             'fold{:02d}_weights.pkl'.format(fold_number)),
                                custom_objects=dict(K=K),
                                compile=False)

    test_ids = pd.read_csv(os.path.join(working,
                                        'test{:02d}.csv'.format(fold_number)),
                           header=None, squeeze=True)

    for file_id in tqdm(test_ids, desc='Predicting'):
        predict_example(model, pump, file_id, working, model_dir)


def evalutron(working, model_dir):

    pump = pickle.load(open(os.path.join(working, 'pump.pkl'), 'rb'))

    jams.util.smkdirs(os.path.join(model_dir, 'predictions'))

    for fold in tqdm(range(5), desc='Fold'):
        test_fold(pump, working, model_dir, fold)


def parse_args(args):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--working', dest='working', type=str,
                        default=os.path.join(os.environ['HOME'],
                                             'working', 'chords'),
                        help='Path to working directory')

    parser.add_argument(dest='model_dir', type=str,
                        help='Path to model directory')

    return parser.parse_args(args)


if __name__ == '__main__':
    params = parse_args(sys.argv[1:])
    evalutron(params.working, params.model_dir)
#
# Things we need
#   - path to where the models live
#   - path to working directory (default='/home/bmcfee/working/chords')
#       - test ids = working + testXX.csv
#       - inputs = working + pump/IDX.npz
#       - outputs = working + predictions/IDX.jams
