#!/bin/env python

import argparse
import pickle
import os
import sys
import numba
from tqdm import tqdm
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

import jams


def mkout(output_dir, filename):

    subidx = os.path.basename(os.path.dirname(filename))

    subdir = os.path.join(output_dir, subidx)

    try:
        jams.util.smkdirs(subdir)
    except:
        pass

    baseidx = os.path.splitext(os.path.basename(filename))[0]

    return os.path.join(subdir, os.path.extsep.join([baseidx, 'npz']))


@numba.jit
def chord_stats(root, alpha=1.0):
    root = root.astype(np.float64)
    N, d = root.shape
    unigram = (alpha * np.ones(d) + root.sum(axis=0)) / (N + d * alpha)
    bigram = alpha * np.ones((d, d))
    p_transition = np.zeros(N)
    all_transitions = 0.0

    for t in range(1, N):
        outer = np.multiply.outer(root[t-1], root[1])
        p_transition[t] = 1 - np.trace(outer)

        # Squash the diagonal and renormalize
        # this way, bigram calculations are conditional on transitioning
        outer -= np.diag(np.diag(outer))
        outer /= (np.sum(outer) + 1e-10)

        all_transitions += p_transition[t]
        bigram += outer * p_transition[t]
    if all_transitions > 0:
        bigram /= all_transitions
    bigram /= (N - 1 + (d**2) * alpha)

    return unigram, bigram, p_transition


def analyze_relative(analysis, tcmodel):

    unigram, bigram, p_trans = chord_stats(analysis['p_root'][0], alpha=1e-3)

    # Drop no-chord states
    unigram = unigram[:12] / unigram[:12].sum()
    bigram = bigram[:12, :12] / bigram[:12, :12].sum()

    # Transitions are row-normalized bigrams
    transition = bigram / bigram.sum(axis=1, keepdims=True)

    # Changes are transitions conditioned on not-self-loop
    changes = transition - np.diag(np.diag(transition))
    changes = changes / changes.sum(axis=1, keepdims=True)

    transition_t = bigram / bigram.sum(axis=0, keepdims=True)
    changes_t = transition_t - np.diag(np.diag(transition_t))
    changes_t = changes_t / changes_t.sum(axis=0, keepdims=True)

    # Compute the weighted average of relative root motions
    centers = tcmodel.predict_proba(analysis['p_pc'].mean(axis=1))[0]

    changes_rel = np.zeros_like(changes)
    changes_t_rel = np.zeros_like(changes_t)
    unigram_rel = np.zeros_like(unigram)

    for i, c in enumerate(centers):
        changes_rel += c * np.roll(changes, -i, axis=(0, 1))
        changes_t_rel += c * np.roll(changes_t, -i, axis=(0, 1))
        unigram_rel += c * np.roll(unigram, -i)

    # Compute the distribution of absolute deviations
    abs_step = np.zeros(12)
    bigram_center = bigram - np.diag(np.diag(bigram))
    bigram_center /= bigram_center.sum()

    for i in range(1, 12):
        e = np.eye(12)
        e = np.roll(e, i, axis=1)

        abs_step[i] = (bigram_center * e).sum()

    return unigram_rel, changes_rel, changes_t_rel, abs_step, centers, p_trans


def predict_example(tcmodel, data_file, output_dir):

    analysis = np.load(data_file)

    unigram, changes, changes_t, abstep, centers, p_trans = analyze_relative(analysis, tcmodel)

    # Save p_root, p_pc to disk
    outfile = mkout(output_dir, data_file)
    np.savez(outfile,
             unigram=unigram,
             changes=changes,
             changes_t=changes_t,
             abstep=abstep,
             centers=centers,
             p_trans=p_trans)


def analyze_data(tcmodel, index_file, output_dir, n_jobs):

    test_files = pd.read_csv(index_file, header=None, squeeze=True)

    Parallel(n_jobs=n_jobs)(delayed(predict_example)(tcmodel, filename, output_dir)
                            for filename in tqdm(test_files, desc='Predicting'))


def evalutron(working, index_file, output_dir, n_jobs):

    tcmodel = pickle.load(open(os.path.join(working, 'tonal_center_model.pkl'), 'rb'))

    analyze_data(tcmodel, index_file, output_dir, n_jobs)


def parse_args(args):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--working', dest='working', type=str,
                        default=os.path.join(os.environ['HOME'],
                                             'working', 'chords'),
                        help='Path to working directory')

    parser.add_argument('--n-jobs', dest='n_jobs', type=int,
                        default=1,
                        help='Number of parallel jobs')

    parser.add_argument(dest='index_file', type=str,
                        help='Path to file index')

    parser.add_argument(dest='output_dir', type=str,
                        help='Path to save outputs')

    return parser.parse_args(args)


if __name__ == '__main__':
    params = parse_args(sys.argv[1:])
    evalutron(params.working,
              params.index_file,
              params.output_dir,
              params.n_jobs)
