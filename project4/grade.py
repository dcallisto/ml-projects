#!/usr/bin/env python3


"""
Official grading script for project 4 of USC's CSCI 467 -- Introduction to ML

Usage:
    > python grade.py data/train_submission.csv data/train_targets.csv

Example Output:
    > Accuracy: 0.1

"""

import sys
import numpy as np

if __name__ == '__main__':
    sub_file = sys.argv[1]
    target_file = sys.argv[2]
    submission = np.loadtxt(sub_file, delimiter=',')
    targets = np.loadtxt(target_file, delimiter=',')

    # Compute accuracy
    pred_targets = np.argmax(targets, axis=1)
    pred_submission = np.argmax(submission, axis=1)
    msg = 'The submission and target files are not the same length.'
    assert pred_targets.shape[0] == pred_submission.shape[0], msg
    acc = pred_targets == pred_submission
    acc = acc.sum() / len(acc)
    print('Accuracy:', round(acc, 4))
