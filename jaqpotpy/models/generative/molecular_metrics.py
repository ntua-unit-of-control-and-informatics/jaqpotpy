from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import pickle
import gzip
import numpy as np
import math

import pkg_resources
from typing import Iterable, Any


NP_file = pkg_resources.resource_filename('jaqpotpy.models.generative.data', 'NP_score.pkl.gz')
SA_file = pkg_resources.resource_filename('jaqpotpy.models.generative.data', 'SA_score.pkl.gz')


NP_model = pickle.load(gzip.open(NP_file))
SA_model = {i[j]: float(i[0]) for i in pickle.load(gzip.open(SA_file)) for j in range(1, len(i))}


def validate_molecule(mol):
    try:
        smiles = Chem.MolToSmiles(mol)
    except Exception as e:
        x = None
    return x is not None and smiles != ''


def novel_score(mols: Iterable[Any], smiles: Iterable[str]):
    valid_mols = []
    valid_smiles = []
    for mol in mols:
        smiles_v = validate_molecule(mol)
        if smiles_v:
            valid_smiles.append(smiles_v)
            valid_mols.append(Chem.MolFromSmiles(smiles_v))
        else:
            valid_mols.append(None)
    ar = np.full((1, len(valid_mols)), True, dtype=bool)
    for index, v_smile in enumerate(valid_smiles):
        if v_smile not in smiles:
            ar[index] = v_smile
    return ar


def novel_mean_score(mols, data):
    valid = novel_score(mols, data)
    return valid.mean()




class MolecularMetrics(object):

    @staticmethod
    def _avoid_sanitization_error(op):
        try:
            return op()
        except ValueError:
            return None

    @staticmethod
    def remap(x, x_min, x_max):
        return (x - x_min) / (x_max - x_min)

    @staticmethod
    def valid_lambda(x):
        try:
            smiles = Chem.MolToSmiles(x)
        except Exception as e:
            x = None
        return x is not None and smiles != ''

    @staticmethod
    def valid_lambda_special(x):
        s = Chem.MolToSmiles(x) if x is not None else ''
        return x is not None and '*' not in s and '.' not in s and s != ''

    @staticmethod
    def valid_scores(mols):
        return np.array(list(map(MolecularMetrics.valid_lambda, mols)), dtype=np.float32)

    @staticmethod
    def valid_filter(mols):
        return list(filter(MolecularMetrics.valid_lambda, mols))

    @staticmethod
    def valid_total_score(mols):
        return np.array(list(map(MolecularMetrics.valid_lambda, mols)), dtype=np.float32).mean()

    @staticmethod
    def novel_scores(mols, data):
        return np.array(
            list(map(lambda x: MolecularMetrics.valid_lambda(x) and Chem.MolToSmiles(x) not in data.smiles, mols)))

    @staticmethod
    def novel_filter(mols, data):
        return list(filter(lambda x: MolecularMetrics.valid_lambda(x) and Chem.MolToSmiles(x) not in data.smiles, mols))

    @staticmethod
    def novel_total_score(mols, data):
        return MolecularMetrics.novel_scores(MolecularMetrics.valid_filter(mols), data).mean()

    @staticmethod
    def unique_scores(mols):
        smiles = list(map(lambda x: Chem.MolToSmiles(x) if MolecularMetrics.valid_lambda(x) else '', mols))
        return np.clip(
            0.75 + np.array(list(map(lambda x: 1 / smiles.count(x) if x != '' else 0, smiles)), dtype=np.float32), 0, 1)

    @staticmethod
    def unique_total_score(mols):
        v = MolecularMetrics.valid_filter(mols)
        s = set(map(lambda x: Chem.MolToSmiles(x), v))
        return 0 if len(v) == 0 else len(s) / len(v)

    @staticmethod
    def natural_product_scores(mols, norm=False):

        # calculating the score
        scores = [sum(NP_model.get(bit, 0)
                      for bit in Chem.rdMolDescriptors.GetMorganFingerprint(mol,
                                                                            2).GetNonzeroElements()) / float(
            mol.GetNumAtoms()) if mol is not None else None
                  for mol in mols]

        # preventing score explosion for exotic molecules
        scores = list(map(lambda score: score if score is None else (
            4 + math.log10(score - 4 + 1) if score > 4 else (
                -4 - math.log10(-4 - score + 1) if score < -4 else score)), scores))

        scores = np.array(list(map(lambda x: -4 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(scores, -3, 1), 0.0, 1.0) if norm else scores

        return scores

    @staticmethod
    def quantitative_estimation_druglikeness_scores(mols, norm=False):
        return np.array(list(map(lambda x: 0 if x is None else x, [
            MolecularMetrics._avoid_sanitization_error(lambda: QED.qed(mol)) if mol is not None else None for mol in
            mols])))

    @staticmethod
    def water_octanol_partition_coefficient_scores(mols, norm=False):
        scores = [MolecularMetrics._avoid_sanitization_error(lambda: Crippen.MolLogP(mol)) if mol is not None else None
                  for mol in mols]
        scores = np.array(list(map(lambda x: -3 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(scores, -2.12178879609, 6.0429063424), 0.0, 1.0) if norm else scores

        return scores

    @staticmethod
    def diversity_scores(mols, data):
        rand_mols = np.random.choice(data, 100)
        fps = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048) for mol in rand_mols]

        scores = np.array(
            list(map(lambda x: MolecularMetrics.__compute_diversity(x, fps) if x is not None else 0, mols)))
        scores = np.clip(MolecularMetrics.remap(scores, 0.9, 0.945), 0.0, 1.0)

        return scores

    @staticmethod
    def __compute_diversity(mol, fps):
        ref_fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
        dist = DataStructs.BulkTanimotoSimilarity(ref_fps, fps, returnDistance=True)
        score = np.mean(dist)
        return score

    @staticmethod
    def drugcandidate_scores(mols, data):

        scores = (MolecularMetrics.constant_bump(
            MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True), 0.210,
            0.945)
                  # + MolecularMetrics.synthetic_accessibility_score_scores(mols,
                  #                                                          norm=True)
                  + MolecularMetrics.novel_scores(
            mols, data) + (1 - MolecularMetrics.novel_scores(mols, data)) * 0.3) / 4

        return scores

    @staticmethod
    def constant_bump(x, x_low, x_high, decay=0.025):
        return np.select(condlist=[x <= x_low, x >= x_high],
                         choicelist=[np.exp(- (x - x_low) ** 2 / decay),
                                     np.exp(- (x - x_high) ** 2 / decay)],
                         default=np.ones_like(x))