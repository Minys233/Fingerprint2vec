"""
Script that trains graph-conv models on Tox21 dataset.
Modified from https://github.com/deepchem/deepchem/blob/master/examples/tox21/tox21_tensorgraph_graph_conv.py

Change log:
1. random splitting for every run
2. simple early stopping policy
3. Output scores on test set
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import os

# np.random.seed(123)
import tensorflow as tf

# tf.random.set_random_seed(123)

import deepchem as dc
from deepchem.molnet import load_tox21
from deepchem.models.graph_models import GraphConvModel, WeaveModel

model_dir = "/home/minys/weave"
os.system(f'rm -r {model_dir}/tox21-featurized')
featurizer = 'Weave' #  'GraphConv'  #
modelcls = WeaveModel #  GraphConvModel #

# Load Tox21 dataset
tox21_tasks, tox21_datasets, transformers = load_tox21(featurizer=featurizer, data_dir=model_dir, save_dir=model_dir)

train_dataset, valid_dataset, test_dataset = tox21_datasets
print(train_dataset.data_dir)
print(valid_dataset.data_dir)

# Fit models
metric = dc.metrics.Metric(
    dc.metrics.roc_auc_score, np.mean, mode="classification")

# Batch size of models
batch_size = 64

model = modelcls(
    len(tox21_tasks), batch_size=batch_size, mode='classification')

# An easy earlystop policy, patience = 5
best = -1
for i in range(10):
    model.fit(train_dataset, nb_epoch=5)
    s = model.evaluate(valid_dataset, [metric], transformers, per_task_metrics=True)
    if s[0]['mean-roc_auc_score'] > best:
        best = s[0]['mean-roc_auc_score']
    else:
        print(f'This is after epoch {(i+1)*5}, average roc = {best:.3f}, early stopping!')
        break

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers, per_task_metrics=True)
valid_scores = model.evaluate(valid_dataset, [metric], transformers, per_task_metrics=True)
test_scores = model.evaluate(test_dataset, [metric], transformers, per_task_metrics=True)

print("Train scores")
print(*train_scores[1]['mean-roc_auc_score'], sep='\t')
print(train_scores[0]['mean-roc_auc_score'])

print("Validation scores")
print(*valid_scores[1]['mean-roc_auc_score'], sep='\t')
print(valid_scores[0]['mean-roc_auc_score'])

print("Test scores")
print(*test_scores[1]['mean-roc_auc_score'], sep='\t')
print(test_scores[0]['mean-roc_auc_score'])

os.system(f'rm -r {model_dir}/tox21-featurized')