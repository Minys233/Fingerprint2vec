from chainer_chemistry.dataset.preprocessors.common \
    import construct_atomic_number_array
from chainer_chemistry.dataset.preprocessors.common import type_check_num_atoms
from chainer_chemistry.dataset.preprocessors.mol_preprocessor \
    import MolPreprocessor

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from gensim.models import Word2Vec
from pathlib import Path


class Node2VecPreprocessor(MolPreprocessor):
    def __init__(self, raduis=1, modelpath='embedding-r1-p0.5-q0.5.model'):
        super(Node2VecPreprocessor, self).__init__()
        self.raduis = raduis
        self.modelpath = modelpath
        if not Path(self.modelpath).is_file():
            raise RuntimeError(f"{self.modelpath} do not exists")
        self.model = Word2Vec.load(modelpath)

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

    def get_input_features(self, mol):
        fps = self.fingerprintSentence(mol)
        feature = np.zeros(self.model.vector_size, dtype=np.float32)
        for f in map(str, fps):
            if f in self.model.wv.vocab:
                feature += self.model.wv[f]
        return feature

if __name__ == '__main__':
    pre = Node2VecPreprocessor()
    print(pre.model.wv.vector_size)


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

    def get_input_features(self, mol):
        fps = self.fingerprintSentence(mol)
        feature = np.zeros(self.model.vector_size, dtype=np.float32)
        for f in map(str, fps):
            if f in self.model.wv.vocab:
                feature += self.model.wv[f]
        return feature
