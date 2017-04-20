#!/usr/bin/env python
'''Model construction and training script'''

import argparse
import os
import sys
from collections import defaultdict
from glob import glob
import six
import pickle

import numpy as np
import pandas as pd
import keras as K

from tqdm import tqdm

from sklearn.model_selection import ShuffleSplit

import jams
import pescador
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
                                             'data', 'eric_chords', 'references_v2'),
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

    parser.add_argument('--static', dest='temporal', action='store_false',
                        help='Use static weighting instead of temporal weighting')

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
                        help='# epochs without improvement to stop')

    parser.add_argument('--reduce-lr', dest='reduce_lr', type=int,
                        default=10,
                        help='# epochs without improvement to reduce learning rate')


    return parser.parse_args(args)


def make_sampler(max_samples, duration, pump, seed):

    n_frames = librosa.time_to_frames(duration,
                                      sr=pump['cqt'].sr,
                                      hop_length=pump['cqt'].hop_length)[0]

    return pump.sampler(max_samples, n_frames, random_state=seed)


def data_sampler(fname, sampler):
    '''Generate samples from a specified npz file'''
    data = np.load(fname)
    d2 = dict(data)
    data.close()
    data = d2
    for datum in sampler(data):
        yield datum


def data_generator(working, tracks, sampler, k, batch_size=32,
                   augmentation=False, weights=None, **kwargs):
    '''Generate a data stream from a collection of tracks and a sampler'''

    seeds = []
    pool_weights = []

    for track in tracks:
        fname = os.path.join(working, 'pump',
                             os.path.extsep.join([track, 'npz']))
        seeds.append(pescador.Streamer(data_sampler, fname, sampler))

        if weights is not None:
            pool_weights.append(weights.loc[track])

        if augmentation:
            for fname in sorted(glob(os.path.join(working, 'pump',
                                                  '{}.*.npz'.format(track)))):
                seeds.append(pescador.Streamer(data_sampler, fname, sampler))
                if weights is not None:
                    pool_weights.append(weights.loc[track])

    # Send it all to a mux
    if not pool_weights:
        pool_weights = None

    mux = pescador.Mux(seeds, k, pool_weights=pool_weights, **kwargs)

    if batch_size == 1:
        return mux
    else:
        return pescador.BufferedStreamer(mux, batch_size)


def keras_tuples(gen, inputs=None, outputs=None):

    if isinstance(inputs, six.string_types):
        if isinstance(outputs, six.string_types):
            # One input, one output
            for datum in gen:
                yield (datum[inputs], datum[outputs])
        else:
            # One input, multi outputs
            for datum in gen:
                yield (datum[inputs], [datum[o] for o in outputs])
    else:
        if isinstance(outputs, six.string_types):
            for datum in gen:
                yield ([datum[i] for i in inputs], datum[outputs])
        else:
            # One input, multi outputs
            for datum in gen:
                yield ([datum[i] for i in inputs],
                       [datum[o] for o in outputs])


def estimate_class_annotation(ann, op, quality_only):
    weights = defaultdict(lambda: 0.0)
    intervals, values = ann.data.to_interval_values()

    for ival, chord in zip(intervals, values):
        chord = op.simplify(chord)

        if quality_only:
            chord = reduce_chord(chord)

        weights[chord] += ival[1] - ival[0]

    return weights, np.max(intervals)


def reduce_chord(c):
    if ':' in c:
        return c[c.rindex(':')+1:]
    else:
        return c


def estimate_class_weights(refs, tracks, op, pseudo=1e-2, quality_only=True):
    seeds = [os.path.join(refs, os.path.extsep.join([track, 'jams']))
             for track in tracks]
    vocab = op.vocabulary()

    if quality_only:
        vocab = set([reduce_chord(c) for c in vocab])

    weights = {k: pseudo for k in vocab}

    total = 0.0

    for jam_in in tqdm(seeds, desc='Estimating class distribution'):
        jam = jams.load(jam_in, validate=False)
        for ann in jam.annotations['chord']:
            weights_i, duration_i = estimate_class_annotation(ann, op,
                                                              quality_only)
            total += duration_i
            for k in weights_i:
                weights[k] += weights_i[k]

    for k in weights:
        weights[k] /= total

    return weights


def weight_track(refs, track, class_weights, op, quality_only=True,
                 aggregate=np.max, temporal=True):
    jam_in = os.path.join(refs, os.path.extsep.join([track, 'jams']))
    jam = jams.load(jam_in, validate=False)

    weight = []
    for ann in jam.annotations['chord']:
        weights_i, duration_i = estimate_class_annotation(ann, op,
                                                          quality_only)

        phat = 0.0
        if not temporal:
            weights_i = set(weights_i.keys())

        for k in weights_i:
            if temporal:
                phat += weights_i[k] / duration_i * np.log(class_weights[k])
            else:
                phat += np.log(class_weights[k]) / len(weights_i)

        weight.append(np.exp(-phat))

    return aggregate(weight)


def weight_tracks(refs, tracks, *args, **kwargs):

    weights = {}
    for track in tqdm(tracks, desc='Estimating track importance weights'):
        weights[track] = weight_track(refs, track, *args, **kwargs)

    return pd.Series(data=weights)


def construct_model(pump, structured):

    INPUTS = 'cqt/mag'

    # Build the input layer
    x = pump.layers()[INPUTS]

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x)

    # First convolutional filter: a single 5x5
    conv1 = K.layers.Convolution2D(1, (5, 5),
                                   padding='same',
                                   activation='relu',
                                   data_format='channels_last')(x_bn)

    # Second convolutional filter: a bank of full-height filters
    conv2 = K.layers.Convolution2D(36, (1, int(conv1.shape[2])),
                                   padding='valid', activation='relu',
                                   data_format='channels_last')(conv1)

    # Squeeze out the frequency dimension
    squeeze = K.layers.Lambda(lambda z: K.backend.squeeze(z, axis=2))(conv2)

    # BRNN layer
    rnn = K.layers.Bidirectional(K.layers.GRU(256,
                                              return_sequences=True))(squeeze)

    if structured:
        # 1: pitch class predictor
        pc = K.layers.Dense(pump.fields['chord_struct/pitch'].shape[1],
                            activation='sigmoid')

        pc_p = K.layers.TimeDistributed(pc, name='chord_pitch')(rnn)

        # 2: root predictor
        root = K.layers.Dense(13, activation='softmax')
        root_p = K.layers.TimeDistributed(root, name='chord_root')(rnn)

        # 3: bass predictor
        bass = K.layers.Dense(13, activation='softmax')
        bass_p = K.layers.TimeDistributed(bass, name='chord_bass')(rnn)

        # 4: merge layer
        codec = K.layers.concatenate([rnn, pc_p, root_p, bass_p])

        p0 = K.layers.Dense(len(pump['chord_tag'].vocabulary()),
                            activation='softmax',
                            bias_regularizer=K.regularizers.l2())

        tag = K.layers.TimeDistributed(p0, name='chord_tag')(codec)

        model = K.models.Model(x, [tag, pc_p, root_p, bass_p])
        OUTPUTS = ['chord_tag/chord',
                   'chord_struct/pitch',
                   'chord_struct/root',
                   'chord_struct/bass']
    else:
        p0 = K.layers.Dense(len(pump['chord_tag'].vocabulary()),
                            activation='softmax',
                            bias_regularizer=K.regularizers.l2())

        tag = K.layers.TimeDistributed(p0, name='chord_tag')(rnn)

        model = K.models.Model(x, [tag])
        OUTPUTS = ['chord_tag/chord']

    return model, INPUTS, OUTPUTS


def make_output_path(working, structured, augmentation, weighted, temporal=True):

    subdir = 'model'
    if structured:
        subdir += '_struct'

    if augmentation:
        subdir += '_aug'

    if weighted:
        subdir += '_weighted'
        if not temporal:
            subdir += '_static'

    outdir = os.path.join(working, subdir)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    return outdir


def score_model(pump, model, idx, working, refs, structured):

    results = {}
    for item in tqdm(idx, desc='Evaluating the model'):
        jam = jams.load('{}/{}.jams'.format(refs, item), validate=False)
        datum = np.load('{}/pump/{}.npz'.format(working, item))['cqt/mag']

        output = model.predict(datum)[0]
        if structured:
            output = output[0]
        ann = pump['chord_tag'].inverse(output)
        results[item] = jams.eval.chord(jam.annotations['chord', 0], ann)

    return pd.DataFrame.from_dict(results, orient='index')[['root', 'thirds',
                                                            'triads', 'tetrads',
                                                            'mirex', 'majmin',
                                                            'sevenths']]


def run_experiment(working, refs, max_samples, duration, structured,
                   augmentation, weighted, temporal, rate,
                   batch_size, epochs, epoch_size, validation_size,
                   early_stopping, reduce_lr, seed):
    '''
    Parameters
    ----------
    working : str
        directory that contains the experiment data (npz)

    refs : str
        directory that contains reference annotations (jams)

    max_samples : int
        Maximum number of samples per streamer

    duration : float
        Duration of training patches

    structured : bool
        Whether or not to use structured training

    augmentation : bool
        Whether to use data augmentation

    weighted : bool
        Whether to use weighted sampling

    temporal : bool
        If using weighting, whether it's static or temporal

    batch_size : int
        Size of batches

    rate : int
        Poisson rate for pescador

    epochs : int
        Maximum number of epoch

    epoch_size : int
        Number of batches per epoch

    validation_size : int
        Number of validation batches

    early_stopping : int
        Number of epochs before early stopping

    reduce_lr : int
        Number of epochs before reducing learning rate

    seed : int
        Random seed
    '''

    # Load the pump
    with open(os.path.join(working, 'pump.pkl'), 'rb') as fd:
        pump = pickle.load(fd)

    # Build the sampler
    sampler = make_sampler(max_samples, duration, pump, seed)

    N_SPLITS = 5
    N_SPLITS = 1

    for split in range(N_SPLITS):

        # Build the model
        model, inputs, outputs = construct_model(pump, structured)

        # Load the training data
        idx_train_ = pd.read_csv(os.path.join(working,
                                              'train{:02d}.csv'.format(split)),
                                 header=None, names=['id'])

        # Split the training data into train and validation
        splitter_tv = ShuffleSplit(n_splits=1, test_size=0.25,
                                   random_state=seed)
        train, val = next(splitter_tv.split(idx_train_))

        idx_train = idx_train_.iloc[train]
        idx_val = idx_train_.iloc[val]

        if weighted:

            chord_weights = estimate_class_weights(refs,
                                                   idx_train['id'].values,
                                                   pump['chord_tag'],
                                                   quality_only=True)

            train_weights = weight_tracks(refs,
                                          idx_train['id'].values,
                                          chord_weights,
                                          pump['chord_tag'],
                                          quality_only=True,
                                          temporal=temporal)
        else:
            train_weights = pd.Series(data={k: 1.0
                                            for k in idx_train['id'].values})

        gen_train = data_generator(working,
                                   train_weights.index, sampler, epoch_size,
                                   augmentation=augmentation,
                                   lam=rate,
                                   batch_size=batch_size,
                                   revive=True,
                                   weights=train_weights,
                                   random_state=seed)

        gen_train = keras_tuples(gen_train(), inputs=inputs, outputs=outputs)

        gen_val = data_generator(working,
                                 idx_val['id'].values, sampler, len(idx_val),
                                 batch_size=batch_size,
                                 revive=True,
                                 random_state=seed)

        gen_val = keras_tuples(gen_val(), inputs=inputs, outputs=outputs)

        loss = {'chord_tag': 'sparse_categorical_crossentropy'}
        metrics = {'chord_tag': 'sparse_categorical_accuracy'}

        if structured:
            loss.update(chord_pitch='binary_crossentropy',
                        chord_root='sparse_categorical_crossentropy',
                        chord_bass='sparse_categorical_crossentropy')
            monitor = 'val_chord_tag_loss'
        else:
            monitor = 'val_loss'

        model.compile(K.optimizers.Adam(), loss=loss, metrics=metrics)

        # Create output path
        output_path = make_output_path(working, structured,
                                       augmentation, weighted,
                                       temporal=temporal)

        # Store the model
        model_spec = K.utils.serialize_keras_object(model)
        with open(os.path.join(output_path,
                               'fold{:02d}_model.pkl'.format(split)),
                  'wb') as fd:
            pickle.dump(model_spec, fd)

        # Construct the weight path
        weight_path = os.path.join(output_path,
                                   'fold{:02d}_weights.pkl'.format(split))

        # Build the callbacks
        cb = []
        cb.append(K.callbacks.ModelCheckpoint(weight_path,
                                              save_best_only=True,
                                              verbose=1,
                                              monitor=monitor))

        cb.append(K.callbacks.ReduceLROnPlateau(patience=reduce_lr,
                                                verbose=1,
                                                monitor=monitor))

        cb.append(K.callbacks.EarlyStopping(patience=early_stopping,
                                            verbose=1,
                                            monitor=monitor))

        # Fit the model
        model.fit_generator(gen_train, epoch_size, epochs,
                            validation_data=gen_val,
                            validation_steps=validation_size,
                            callbacks=cb)

        ###
        # Now test the model
        # Load the best weights
        model.load_weights(weight_path)

        # Load the testing data
        idx_test = pd.read_csv(os.path.join(working,
                                            'test{:02d}.csv'.format(split)),
                               header=None, names=['id'])

        test_scores = score_model(pump, model, idx_test['id'], working, refs, structured)

        output_scores = os.path.join(output_path,
                                     'fold{:02d}_test.csv'.format(split))
        test_scores.to_csv(output_scores)


if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])

    print(params)

    run_experiment(params.working, params.refs,
                   params.max_samples, params.duration,
                   params.structured, params.augmentation, params.weighted, params.temporal,
                   params.rate,
                   params.batch_size,
                   params.epochs, params.epoch_size,
                   params.validation_size,
                   params.early_stopping,
                   params.reduce_lr,
                   params.seed)
