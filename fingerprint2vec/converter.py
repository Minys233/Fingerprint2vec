from pathlib import Path
from gensim.models import Word2Vec
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


class Embedding:
    """
    This class handles trained embedding models.
    You can use this class for converting molecules to vectors or sequences of vectors.
    """
    def __init__(self, modelpath):
        """
        Load and manage pre-trained model.
        :param modelpath: the pre-trained model. The name contain [...]-r[number]-p[number]-q[number].[ext]
        for extracting information, eg "embedding-r1-p0.5-q0.5.model"
        """
        self.modelpath = Path(modelpath)
        if not self.modelpath.is_file():
            raise FileNotFoundError(f"{self.modelpath} does not exist")
        self.model = Word2Vec.load(modelpath)
        elems = self.modelpath.name.rsplit('.', 1)[0].split('-')[-3:]
        self.r = int(elems[0][1:])
        self.p = float(elems[1][1:])
        self.q = float(elems[2][1:])
        print(f'{len(self.model.wv.index2word)} sub-structures with dimension {self.model.vector_size}')

    def substructure(self, mol):
        """
        Extract substructures from molecule, using Morgan fingerprint
        :param mol: A Chem.Mol instance
        :return: A list of string indicating substructures
        """
        info = dict()
        # key: fp_str, val: ((atom_idx, radius),...,)
        AllChem.GetMorganFingerprint(mol, self.r, bitInfo=info)
        atomidx2fp = [[None for _ in range(self.r + 1)] for __ in range(mol.GetNumAtoms())]
        for fp_int, frag in info.items():
            for atom_idx, r in frag:
                atomidx2fp[atom_idx][r] = fp_int
        sentence = list()
        for atom_idx in range(mol.GetNumAtoms()):
            for r in range(self.r + 1):
                if atomidx2fp[atom_idx][r]:
                    sentence.append(atomidx2fp[atom_idx][r])
        return list(map(str, sentence))

    def embed(self, smi_or_mol):
        """
        Calculate the embedding of one molecule.
        :param smi_or_mol: SMILES string or Chem.Mol instance
        :return: One numpy array
        """
        if not isinstance(smi_or_mol, Chem.Mol):
            mol = Chem.MolFromSmiles(smi_or_mol)
        else:
            mol = smi_or_mol
        wv = self.model.wv
        sentence = self.substructure(mol)
        vec = np.zeros(self.model.vector_size)
        for fp in sentence:
            if fp in wv.vocab:
                vec += wv[fp]
        return vec

    def embed_batch(self, smi_or_mol_lst):
        """
        Do the same embedding job in batch
        :param smi_or_mol_lst: SMILES iterator or Chem.Mol iterator
        :return: A numpy array, with shape (N, self.model.vector_size)
        """
        mol_lst = []
        for i in smi_or_mol_lst:
            if not isinstance(i, Chem.Mol):
                mol_lst.append(Chem.MolFromSmiles(i))
            else:
                mol_lst.append(i)
        vecs = np.zeros([len(mol_lst), self.model.vector_size])
        wv, vocab = self.model.wv, self.model.wv.vocab
        for idx, mol in enumerate(mol_lst):
            for fp in self.substructure(mol):
                if fp in vocab:
                    vecs[idx] += wv[fp]
        return vecs


if __name__ == '__main__':
    # test
    embd = Embedding('../models/embedding-r1-p0.5-q0.5.model')
    print(embd.embed('CCCCC').shape)
    print(embd.embed_batch(['c1ccccc1', 'CC', 'CCCCC']).shape)
