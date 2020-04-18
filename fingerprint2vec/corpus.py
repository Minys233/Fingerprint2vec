import argparse
from collections import defaultdict
import random
import sys

from tqdm import tqdm
from rdkit.Chem import AllChem


def pathes2fp(mol, fp_raduis, pathes):
    bitinfo, atom2fp = dict(), defaultdict(list)
    AllChem.GetMorganFingerprint(mol, fp_raduis, bitInfo=bitinfo)
    for fp, positions in bitinfo.items():
        for atmidx, r in positions:
            atom2fp[atmidx].append(fp)
    fppathes = []
    for path in pathes:
        fppath = []
        for atmidx in path:
            fplst = atom2fp[atmidx]
            random.shuffle(fplst)
            fppath += fplst
        fppathes.append(fppath)
    return fppathes


def Node2Vec(smi, maxlength, fp_raduis, p, q):
    # random walk from every atoms
    mol = AllChem.MolFromSmiles(smi)
    dm = AllChem.GetDistanceMatrix(mol)
    nb = [list(map(lambda x: x.GetIdx(), mol.GetAtomWithIdx(i).GetNeighbors())) for i in range(mol.GetNumAtoms())]
    startatoms = list(range(mol.GetNumAtoms()))
    # path = Parallel(n_jobs=-1)(delayed(Node2VecHelper)(mol, dm, nb, i, maxlength, p, q) for i in startatoms)
    path = [Node2VecHelper(mol, dm, nb, i, maxlength, p, q) for i in startatoms]
    # convert ints to fingerprints
    fppath = pathes2fp(mol, fp_raduis, path)
    return path, fppath


def Node2VecHelper(mol, dm, nb, idx, length, p, q, epsilon=0.5):
    path = [idx, ]
    startatom = mol.GetAtomWithIdx(path[-1])
    neighbors = list(map(lambda x: x.GetIdx(), startatom.GetNeighbors()))
    path.append(random.choice(neighbors))
    coef = [1 if i not in path else 1*epsilon for i in range(mol.GetNumAtoms())]
    for i in range(length-2):
        neighbors = nb[path[-1]]
        t = path[-2]
        alpha = []
        for xi in neighbors:
            d = int(dm[t, xi])
            if xi in path: coef[xi] *= epsilon
            if d == 0: alpha.append(1/p * coef[xi])  # return to father, BFS
            elif d == 1: alpha.append(1 * coef[xi])  # rare case! only happen when 3-ring
            elif d == 2: alpha.append(1/q * coef[xi])  # deeper, DFS
        weight = alpha
        path.append(random.choices(neighbors, weights=weight)[0])
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help="input file containing smiles string")
    parser.add_argument('output', type=str, help="output file containing fingerprint strings")
    parser.add_argument('--maxlen', dest='maxlen', type=int, help="max length of random walk (not max length for sentence)")
    parser.add_argument('--raduis', dest='raduis', type=int, help="max raduis for morgan fingerprint")
    parser.add_argument('--p', dest='p', type=float, help="p in Node2vec")
    parser.add_argument('--q', dest='q', type=float, help="q in Node2vec")
    args = parser.parse_args()
    if args.output == '-':
        infile, outfile = open(args.input, 'r'), sys.stdout
    else:
        infile, outfile = open(args.input, 'r'), open(args.output, 'w')
    for smi in tqdm(infile.readlines()):
        path, fppath = Node2Vec(smi, args.maxlen, args.raduis, args.p, args.q)
        for fpp in fppath:
            print(*fpp, file=outfile)



