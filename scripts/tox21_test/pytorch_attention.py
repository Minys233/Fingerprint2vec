import math
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split

from gensim.models import Word2Vec
import numpy as np
from rdkit.Chem import AllChem
import pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('modelpath', type=str)
parser.add_argument('--pad_to', type=int, default=80)
parser.add_argument('colname', type=str, choices=['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
                  'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
                  'SR-MMP', 'SR-p53'])
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--saveto', type=str, default='./')
parser.add_argument('epoch', type=int, default=20)

args = parser.parse_args()
modelpath = args.modelpath
pad_to = args.pad_to
epoch = args.epoch
colname = args.colname
saveto = Path(args.saveto)
if not saveto.is_dir():
    saveto.mkdir(exist_ok=True, parents=True)
device = args.device
lossname = 'setme'

class AttentionRNN(nn.Module):
    def __init__(self, emb_dim=300, hid_dim=150, r=10, d=8, num_rnn_layer=1, drop_rate=0.5, bidirectional=True):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.hid_dim = int(hid_dim)
        self.num_direction = 2 if bidirectional else 1
        self.num_rnn_layers = int(num_rnn_layer)
        self.droprate = float(drop_rate)
        self.r, self.d = int(r), int(d)
        self.rnn = nn.GRU(self.emb_dim, self.hid_dim,
                          self.num_rnn_layers, bidirectional=True, batch_first=True)
        # simple self attention
        self.atten_W1 = nn.Parameter(torch.Tensor(self.d, self.num_direction * self.hid_dim))
        nn.init.kaiming_uniform_(self.atten_W1, a=math.sqrt(5))
        self.atten_W2 = nn.Parameter(torch.Tensor(self.r, self.d))
        nn.init.kaiming_uniform_(self.atten_W2, a=math.sqrt(5))
        self.linear_squeeze = nn.Linear(self.num_direction*self.hid_dim, 1)
        self.dropout = nn.Dropout(self.droprate)
        self.linear_output = nn.Linear(self.r, 1)

    def forward(self, input_seqs, nonpad):
        """
        :param input_seqs: shape (batchsize, seqlen, emb_dim)
        :return:
        """
        output, h_n = self.rnn(input_seqs)
        # output: (batchsize, seqlen, num_directions * hidden_size),
        # h_n: (batchsize, num_rnn_layer * num_directions, hidden_size)
        o1 = torch.matmul(self.atten_W1, output.transpose(1,2)) # W1 (d, num_directions * hidden_size)
        # o1: (batchsize, d, seqlen)
        o1 = torch.tanh(o1)
        o2 = torch.matmul(self.atten_W2, o1)  # W2 (r, d)
        # o2: (batchsize, r, seqlen) # r attentions

        for idx in range(nonpad.shape[0]):
            if nonpad[idx] < pad_to:
                o2[idx, :, nonpad[idx]:] = -100

        A = F.softmax(o2, dim=2)

        # A: (batchsize, r, seqlen)
        M = torch.matmul(A, output)

        # M: (batchsize, r, num_directions * hidden_size)
        y = self.dropout(M)
        y = self.linear_squeeze(y).squeeze()
        y = self.linear_output(y)
        return y, A


class RMSELoss(nn.MSELoss):
    def __init__(self, *args, **kwargs):
        super(RMSELoss, self).__init__(*args, **kwargs)

    def forward(self, input, target):
        return torch.sqrt(super(RMSELoss, self).forward(input, target))


class MoleculeDataset(Dataset):
    def __init__(self, modelpath, raduis, molecules, labels, pad_to=40):
        self.word2vec = Word2Vec.load(modelpath)
        # print(f"Load {modelpath}, with {len(self.word2vec.wv.vocab)} words.")
        self.raduis = raduis
        self.molecules = molecules
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.pad_to = pad_to
        assert len(molecules) == self.labels.shape[0]

        pos_idx = self.labels > 0.5

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, idx):

        sent = self.fingerprintSentence(self.molecules[idx])
        arr = np.zeros([self.pad_to, self.word2vec.wv.vector_size], dtype=np.float32)
        effictive_count = 0
        for c in sent:
            if c in self.word2vec.wv.vocab:
                arr[effictive_count] = self.word2vec.wv[c]
                effictive_count += 1
            if effictive_count == self.pad_to:
                break
        arr = torch.from_numpy(arr)
        sample = {'seq': arr, 'label': self.labels[idx], 'nonpad': torch.tensor(min(len(sent), self.pad_to), dtype=torch.int64)}
        return sample

    def fingerprintSentence(self, mol: [AllChem.Mol, str]):
        info = dict()
        if isinstance(mol, str):
            mol = AllChem.MolFromSmiles(mol)
        # key: fp_str, val: ((atom_idx, radius),...,)
        AllChem.GetMorganFingerprint(mol, self.raduis, bitInfo=info)
        atomidx2fp = [[None for _ in range(self.raduis + 1)] for __ in range(mol.GetNumAtoms())]
        for fp_int, frag in info.items():
            for atom_idx, r in frag:
                atomidx2fp[atom_idx][r] = fp_int
        sentence = list()
        for atom_idx in range(mol.GetNumAtoms()):
            for r in range(self.raduis + 1):
                if atomidx2fp[atom_idx][r]:
                    sentence.append(atomidx2fp[atom_idx][r])
        return list(map(str, sentence))

def load_split(csv_path, smi_col, label_col, splits):
    df = pd.read_csv(csv_path)
    smis = df[smi_col].to_numpy()
    mols = np.array([AllChem.MolFromSmiles(i) for i in df[smi_col]])
    labels = df[label_col].to_numpy()
    valididx = ~np.isnan(labels)
    smis, mols, labels = smis[valididx], mols[valididx], labels[valididx]
    assert len(mols) == len(labels)
    assert np.isclose(np.sum(splits), 1.0)
    num_data = len(mols)
    index = np.random.permutation(num_data)
    mols, labels = mols[index], labels[index]
    part1 = int(num_data * splits[0])
    part2 = int(num_data * (splits[0]+splits[1]))
    mols_train, labels_train = mols[:part1], labels[:part1]
    mols_valid, labels_valid = mols[part1:part2], labels[part1:part2]
    mols_test, labels_test = mols[part2:], labels[part2:]
    assert len(mols_train) == len(labels_train)
    assert len(mols_valid) == len(labels_valid)
    assert len(mols_test) == len(labels_test)

    return (mols_train, labels_train), (mols_valid, labels_valid), (mols_test, labels_test)


def train_model(**kwargs):
    global colname, lossname, epoch, modelpath, pad_to, device, saveto
    train, valid, test = load_split('tox21.csv',
                                    'smiles',
                                    colname,
                                    [0.8, 0.1, 0.1])

    trainset = MoleculeDataset(modelpath, 1, train[0], train[1], pad_to=pad_to)
    validset = MoleculeDataset(modelpath, 1, valid[0], valid[1], pad_to=pad_to)
    testset = MoleculeDataset(modelpath, 1, test[0], test[1], pad_to=pad_to)

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    validloader = DataLoader(validset, batch_size=128, shuffle=False)
    testloader = DataLoader(testset, batch_size=128, shuffle=False)

    model = AttentionRNN(**kwargs)
    model.to(device)
    # decrease performance!
    # weight = torch.tensor([(trainset.labels.shape[0]-trainset.labels.sum())/trainset.labels.sum()])
    # print(f"Weight for positive classes: {weight}")
    criterion = nn.BCEWithLogitsLoss()
    lossname = criterion.__class__.__name__
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    valid_best = math.inf
    for e in range(epoch):
        # print(f'Training epoch {e}')
        epoch_loss = 0
        for step, data in enumerate(trainloader):
            seq, label, nonpad = data['seq'].to(device), data['label'].to(device), data['nonpad'].to(device)
            optimizer.zero_grad()
            output, atten = model(seq, nonpad)
            loss = criterion(output.squeeze(), label.squeeze())
            loss.backward()
            optimizer.step()
            # print(f"    Step -- {lossname}: {loss.item()}")
            epoch_loss += len(label) * loss.item()
        model.eval()
        with torch.no_grad():
            eval_loss = 0
            for data in validloader:
                seq, label, nonpad = data['seq'].to(device), data['label'].to(device), data['nonpad'].to(device)
                output, atten = model(seq, nonpad)
                loss = criterion(output.squeeze(), label.squeeze())
                eval_loss += loss.item() * len(seq)
            eval_loss /= len(validset)
            print(f"\rTraining epoch {e}    Validset {lossname}: {eval_loss}          ", end='')
            if eval_loss <= valid_best:
                valid_best = eval_loss
                torch.save({'model': model.state_dict(), 'epoch': e,
                            'metric': valid_best, 'modelpath': modelpath},
                           saveto / f'attention_rnn_model_{colname}.pth')
        model.train()
        epoch_loss /= len(trainset)
        # print(f"Epoch train {lossname}: {epoch_loss}")
    return evaluate_model(testloader, **kwargs)


def evaluate_model(test_loader, **kwargs):
    global colname, device, lossname, saveto
    model = AttentionRNN(**kwargs)
    criterion = nn.BCEWithLogitsLoss()
    ckpt = torch.load(saveto / f'attention_rnn_model_{colname}.pth')
    model.load_state_dict(ckpt['model'])
    print(f'{ckpt["metric"]}')
    model.to(device)
    model.eval()
    test_loss = 0
    pred, truth = [], []
    with torch.no_grad():
        for data in test_loader:
            seq, label, nonpad = data['seq'].to(device), data['label'].to(device), data['nonpad'].to(device)
            output, atten = model(seq, nonpad)
            loss = criterion(output.squeeze(), label.squeeze())
            pred.append(output.squeeze().cpu().numpy())
            truth.append(label.cpu().numpy())
            test_loss += loss.item() * len(seq)
        test_loss /= len(test_loader.dataset)
    pred, truth = np.concatenate(pred), np.concatenate(truth)
    assert pred.shape == truth.shape
    roc = roc_auc_score(truth, pred)
    # print(f"Test {lossname}: {test_loss}")
    # print(f"ROC-AUC: ", roc)
    return roc


if __name__ == '__main__':
    # NR-AR
    print()
    roc = train_model(hid_dim=200, r=20, d=40)
    print()
    print(roc)
    exit(0)
	
	from pyGPGO.covfunc import matern32, squaredExponential
	from pyGPGO.acquisition import Acquisition
	from pyGPGO.surrogates.GaussianProcess import GaussianProcess
	from pyGPGO.GPGO import GPGO
    cov = squaredExponential()
    gp = GaussianProcess(cov)
    acq = Acquisition(mode='ExpectedImprovement')
    param = {
        'hid_dim': ('int', [100, 300]),
             'r': ('int', [5, 30]),
             'd': ('int', [5, 100]),
            }

    gpgo = GPGO(gp, acq, train_model, param, n_jobs=2)
    gpgo.run(max_iter=100)




