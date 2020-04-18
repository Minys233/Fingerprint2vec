#!/usr/bin/env python
from __future__ import print_function

import argparse
import os
import pickle as pkl

import cupy
import chainer
from chainer.dataset import convert
from chainer import configuration
from chainer import function, functions
from chainer.iterators import SerialIterator
from chainer.training.extensions import Evaluator
from chainer_chemistry.training.extensions.roc_auc_evaluator import ROCAUCEvaluator  # NOQA
from chainer_chemistry.training.extensions.prc_auc_evaluator import PRCAUCEvaluator
# Proposed by Ishiguro
# ToDo: consider go/no-go with following modification
# Re-load the best-validation score snapshot using serializers
from chainer import serializers

from chainer_chemistry.dataset.converters import converter_method_dict
from chainer_chemistry.datasets import NumpyTupleDataset
from chainer_chemistry.datasets.molnet.molnet_config import molnet_default_config  # NOQA
from chainer_chemistry.models.prediction import Classifier
from chainer_chemistry.models.prediction import Regressor
from chainer_chemistry.utils import save_json

# These import is necessary for pickle to work
from chainer_chemistry.links.scaler.standard_scaler import StandardScaler  # NOQA
from chainer_chemistry.models.prediction.graph_conv_predictor import GraphConvPredictor  # NOQA
from molnet.train_molnet import dataset_part_filename
from molnet.train_molnet import download_entire_dataset


def parse_arguments():
    # Lists of supported preprocessing methods/models.
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn', 'relgcn',
                   'relgat', 'gin', 'gnnfilm', 'megnet',
                   'nfp_gwm', 'ggnn_gwm', 'rsgcn_gwm', 'gin_gwm', 'node2vec']
    scale_list = ['standardize', 'none']
    dataset_names = list(molnet_default_config.keys())

    # Set up the argument parser.
    parser = argparse.ArgumentParser(description='Prediction on Molnet.')
    parser.add_argument('--dataset', '-d', type=str, choices=dataset_names,
                        default='bbbp',
                        help='name of the dataset that training is run on')
    parser.add_argument('--method', '-m', type=str, choices=method_list,
                        help='method name', default='nfp')
    parser.add_argument('--label', '-l', type=str, default='',
                        help='target label for regression; empty string means '
                        'predicting all properties at once')
    parser.add_argument('--scale', type=str, choices=scale_list,
                        help='label scaling method', default='none')
    parser.add_argument(
        '--device', type=str, default='-1',
        help='Device specifier. Either ChainerX device specifier or an '
             'integer. If non-negative integer, CuPy arrays with specified '
             'device id are used. If negative integer, NumPy arrays are used')
    parser.add_argument('--in-dir', '-i', type=str, default='result',
                        help='directory to load model data from')
    parser.add_argument('--num-data', type=int, default=-1,
                        help='amount of data to be parsed; -1 indicates '
                        'parsing all data.')

    parser.add_argument('--modelpath', type=str, help='Node2vec or Word2vec model path', default=None)
    parser.add_argument('--datapath', type=str, help='input data dir', default='input')

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Set up some useful variables that will be used later on.
    dataset_name = args.dataset
    method = args.method
    num_data = args.num_data

    if method == 'node2vec':
        fname = args.modelpath.split('/')[-1].rsplit('.', 1)[0]
        r, p, q = fname.split('-')[-3:]
        r, p, q = int(r[1:]), float(p[1:]), float(q[1:])
        if args.label:
            print(args.modelpath, f"{args.dataset}_{args.label}")
        else:
            print(args.modelpath, args.dataset)

    if args.label:
        labels = args.label
        if method == 'node2vec':
            cache_dir = os.path.join(args.datapath, '{}_{}_r{}_p{}_q{}_{}'.format(dataset_name,
                                                                            method, r, p, q, labels))
        else:
            cache_dir = os.path.join(args.datapath, '{}_{}_{}'.format(dataset_name,
                                                                method, labels))
    else:
        labels = None
        if method == 'node2vec':
            cache_dir = os.path.join(args.datapath, '{}_{}_r{}_p{}_q{}_all'.format(dataset_name,
                                                                            method, r, p, q))
        else:
            cache_dir = os.path.join(args.datapath, '{}_{}_all'.format(dataset_name,
                                                                method))

    # Load the cached dataset.
    filename = dataset_part_filename('test', num_data)
    path = os.path.join(cache_dir, filename)
    if os.path.exists(path):
        # print('Loading cached dataset from {}.'.format(path))
        test = NumpyTupleDataset.load(path)
    else:
        # _, _, test = download_entire_dataset(dataset_name, num_data, labels,
        #                                      method, cache_dir, modelpath=args.modelpath)
        raise RuntimeError('dataset must be set, same as train')

    # Model-related data is stored this directory.
    model_dir = os.path.join(args.in_dir, os.path.basename(cache_dir))

    model_filename = {'classification': 'classifier.pkl',
                      'regression': 'regressor.pkl'}
    task_type = molnet_default_config[dataset_name]['task_type']
    model_path = os.path.join(model_dir, model_filename[task_type])
    # print("model_path=" + model_path)
    # print('Loading model weights from {}...'.format(model_path))


    device = chainer.get_device(args.device)
    if task_type == 'classification':
        model = Classifier.load_pickle(model_path, device=device)
    elif task_type == 'regression':
        model = Regressor.load_pickle(model_path, device=device)
    else:
        raise ValueError('Invalid task type ({}) encountered when processing '
                         'dataset ({}).'.format(task_type, dataset_name))

    # Re-load the best-validation score snapshot
    serializers.load_npz(os.path.join(
        model_dir, "best_val_" + model_filename[task_type]), model)

    # Run an evaluator on the test dataset.
    # print('Evaluating...')
    if method == 'node2vec':
        converter = converter_method_dict['nfp']  # concat_mols
    else:
        converter = converter_method_dict[method]

    test_iterator = SerialIterator(test, 16, repeat=False, shuffle=False)


    if task_type == 'regression' and args.scale == 'standardize':
        scaler = pkl.load(open(os.path.join(cache_dir, 'standatdize_scaler.pkl'), 'rb'))
        if dataset_name == 'qm9':
            errfn = functions.mean_absolute_error
        else:
            errfn = functions.mean_squared_error
        total_err = 0
        for batch in test_iterator:
            with configuration.using_config('train', False):
                in_arrays = convert._call_converter(converter, batch, device)
                with function.no_backprop_mode():
                    pred = cupy.asnumpy(model.predictor(in_arrays[0]).array)
                    y = cupy.asnumpy(in_arrays[1])

                    pred = scaler.inverse_transform(pred)

                    total_err += errfn(pred, y).item() * len(y)
        xp = cupy.get_array_module(in_arrays[0])
        total_err = xp.array(total_err) / len(test)
        if dataset_name != 'qm9':
            total_err = functions.sqrt(total_err)
            eval_result = {'main/loss': 0.0, 'main/RMSE': total_err.item()}
        else:
            eval_result = {'main/loss': 0.0, 'main/MAE': total_err.item()}
    else:
        eval_result = Evaluator(test_iterator, model, converter=converter, device=device)()
    print('Evaluation result: ', eval_result)

    # Add more stats
    if task_type == 'regression':
        # loss = cuda.to_cpu(numpy.array(eval_result['main/loss']))
        # eval_result['main/loss'] = loss

        # convert to native values..
        for k, v in eval_result.items():
            eval_result[k] = float(v)

    elif task_type == "classification":
        # For Classifier, we do not equip the model with ROC-AUC evalation
        # function. use separate ROC-AUC Evaluator
        if dataset_name in ['muv', 'pcba']:
            prcauc_result = PRCAUCEvaluator(
                test_iterator, model, converter=converter, device=device,
                eval_func=model.predictor, name='test', ignore_labels=-1)()
            print('PRCAUC Evaluation result: ', prcauc_result)
            save_json(os.path.join(model_dir, 'prcauc_result.json'), prcauc_result)
        else:
            rocauc_result = ROCAUCEvaluator(
                test_iterator, model, converter=converter, device=device,
                eval_func=model.predictor, name='test', ignore_labels=-1)()
            print('ROCAUC Evaluation result: ', rocauc_result)
            save_json(os.path.join(model_dir, 'rocauc_result.json'), rocauc_result)

    else:
        print('[WARNING] unknown task_type {}.'.format(task_type))

    # Save the evaluation results.
    save_json(os.path.join(model_dir, 'eval_result.json'), eval_result)


if __name__ == '__main__':
    main()
