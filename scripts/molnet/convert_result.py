import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output1')
parser.add_argument('output2')

args = parser.parse_args()

lines = open(args.input).readlines()
assert len(lines) % 2 == 0

headers = ['r', 'p', 'q', 'dataset', 'metric', 'score']
classification, regression = [], []

for idx in range(0, len(lines), 2):
    modelpath, dataset = lines[idx].split()
    r, p, q = modelpath.split('/')[-1].rsplit('.', 1)[0].split('-')[2:5]
    r, p, q = int(r[1:]), float(p[1:]), float(q[1:])
    print(modelpath)
    print(f"r={r}, p={p}, q={q}")
    if lines[idx + 1].startswith('ROCAUC'):
        roc = float(lines[idx + 1].split(':')[-1][:-2])
        classification.append([r, p, q, dataset, 'ROC-AUC', roc])
    elif lines[idx + 1].startswith('PRCAUC'):
        prc = float(lines[idx + 1].split(':')[-1][:-2])
        classification.append([r, p, q, dataset, 'PRC-AUC', prc])
    else:
        metric, score = lines[idx + 1].split(':')[-2:]
        metric = metric.split(',')[-1].split('/')[-1][:-1]
        score = float(score[:-2])
        regression.append([r, p, q, dataset, metric, score])

classification = pd.DataFrame(classification, columns=headers)
classification.to_csv(args.output1, index=False)
regression = pd.DataFrame(regression, columns=headers)
regression.to_csv(args.output2, index=False)

