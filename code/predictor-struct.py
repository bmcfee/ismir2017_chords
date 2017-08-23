#!/bin/env python

import argparse
import pickle
import os
import sys
import keras as K
from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy.stats

import jams
from librosa import time_to_frames
import mir_eval


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

SEMITONE_TO_SCALE_DEGREE = ['1', 'b2', '2', 'b3', '3', '4', 'b5', '5', 'b6', '6', 'b7', '7']


def preds_to_inversions(pump, tag_pred, bass_pred):

    pump_op = pump['chord_tag']
    ann = pump['chord_tag'].inverse(tag_pred)

    for obs in ann.pop_data():
        value = obs.value

        if obs.value not in ('N', 'X'):
            # 1. Convert (time, time+duration) to frames
            start, end = time_to_frames([obs.time, obs.time + obs.duration],
                                        sr=pump_op.sr, hop_length=pump_op.hop_length)

            # 2. G-mean aggregate bass_pred at the frame level
            mean_bass = scipy.stats.gmean(bass_pred[start:end+1])

            # 3. Find the argmax of the aggregated frames
            bass_pc = np.argmax(mean_bass)

            # 4. map argmax to scale degree
            root_pc, pitches, _ = mir_eval.chord.encode(obs.value)

            bass_rel = 0
            if bass_pc < 12:
                bass_rel = np.mod(bass_pc - root_pc, 12)

            if bass_rel and pitches[bass_rel]:
                value = '{}/{}'.format(value, SEMITONE_TO_SCALE_DEGREE[bass_rel])

        # 5. Re-insert to ann
        ann.append(time=obs.time, duration=obs.duration,
                   value=value,
                   confidence=obs.confidence)
    return ann


def predict_example(model, pump, file_id, working, model_dir):

    data = dict(np.load(os.path.join(working, 'pump', '{}.npz'.format(file_id))))
    p_tag, p_pc, p_root, p_bass = model.predict(data['cqt/mag'])
    ann = preds_to_inversions(pump, p_tag[0], p_bass[0])
    J = jams.JAMS()
    J.file_metadata.identifiers.track_id = file_id
    J.file_metadata.duration = ann.data[-1].time + ann.data[-1].duration
    J.annotations.append(ann)
    J.save(os.path.join(model_dir, 'predict_struct', '{}.jams'.format(file_id)))


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

    jams.util.smkdirs(os.path.join(model_dir, 'predict_struct'))

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
