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
        x = Chem.MolToSmiles(mol)
    except Exception as e:
        x = None
    return x


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
            valid_smiles.append(None)
    ar = np.full((1, len(valid_mols)), True, dtype=bool)
    for index, v_smile in enumerate(valid_smiles):
        if v_smile not in smiles:
            ar[0][index] = v_smile
    return ar[0]


def novel_mean_score(mols, data):
    valid = novel_score(mols, data)
    return valid.mean()


def valid_mean(mols):
    valids = []
    for m in mols:
        try:
            smiles = Chem.MolToSmiles(m)
            valids.append(1)
        except Exception as e:
            valids.append(0)
    return np.array(valids).mean()


def valids(mols):
    # return x is not None and Chem.MolToSmiles(x) != ''
    valids = []
    try:
        for m in mols:
            try:
                smiles = Chem.MolToSmiles(m)
                valids.append(m)
            except Exception as e:
                valids.append(None)
    except TypeError as e:
        try:
            smiles = Chem.MolToSmiles(mols)
            valids.append(mols)
        except Exception as e:
            valids.append(None)
    return valids


def compute_SAS(mol):
    if mol is None:
        return 0
    fp = Chem.rdMolDescriptors.GetMorganFingerprint(mol, 2)
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    # for bitId, v in fps.items():
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += SA_model.get(sfp, -4) * v
    try:
        score1 /= nf
    except ZeroDivisionError as e:
        return 0

    # features score
    nAtoms = mol.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(
        mol, includeUnassigned=True))
    ri = mol.GetRingInfo()
    nSpiro = Chem.rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgeheads = Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms ** 1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.

    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - \
             spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore


def _remap(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


def synthetic_accessibility_score_scores(mols, norm=False):
    scores = [compute_SAS(mol) if mol is not None else None for mol in mols]
    scores = np.array(list(map(lambda x: 0 if x is None else x, scores)))
    scores = np.clip(_remap(scores, 5, 1.5), 0.0, 1.0) if norm else scores
    return scores


def natural_products_score(mols, norm=False):
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
    scores = np.clip(_remap(scores, -3, 1), 0.0, 1.0) if norm else scores

    return scores


def _avoid_sanitization_error(op):
    try:
        return op()
    except ValueError:
        return None


def quantitative_estimation_druglikeness_scores(mols, norm=True):
    return np.array(list(map(lambda x: 0 if x is None else x, [
        _avoid_sanitization_error(lambda: QED.qed(mol)) if mol is not None else None for mol in
        mols])))


def water_octanol_partition_coefficient_scores(mols, norm=False):
    scores = [_avoid_sanitization_error(lambda: Crippen.MolLogP(mol)) if mol is not None else None
              for mol in mols]
    scores = np.array(list(map(lambda x: -3 if x is None else x, scores)))
    scores = np.clip(_remap(scores, -2.12178879609, 6.0429063424), 0.0, 1.0) if norm else scores
    return scores


def _compute_diversity(mol, fps):
    ref_fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
    dist = DataStructs.BulkTanimotoSimilarity(ref_fps, fps, returnDistance=True)
    score = np.mean(dist)
    return score


def diversity_scores(mols, data):
    rand_mols = np.random.choice(data, 100)
    fps = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048) for mol in rand_mols]

    scores = np.array(
        list(map(lambda x: _compute_diversity(x, fps) if x is not None else 0, mols)))
    scores = np.clip(_remap(scores, 0.9, 0.945), 0.0, 1.0)
    return scores


def unique_total_score(mols):
    v = valids(mols)
    s = set()
    for mol in v:
        try:
            smile = Chem.MolToSmiles(mol)
            s.add(smile)
        except Exception as e:
            continue
    # s = set(map(lambda x: Chem.MolToSmiles(x), v))
    return 0 if len(v) == 0 else len(s) / len(v)


def _constant_bump(x, x_low, x_high, decay=0.025):
    return np.select(condlist=[x <= x_low, x >= x_high],
                     choicelist=[np.exp(- (x - x_low) ** 2 / decay),
                                 np.exp(- (x - x_high) ** 2 / decay)],
                     default=np.ones_like(x))


def novel_scores(mols, data):
    smiles = []
    for mol in mols:
        try:
            sm = Chem.MolToSmiles(mol)
            smiles.append(sm)
        except Exception as e:
            smiles.append(None)
    novel_l = []
    for smile in smiles:
        if smile not in data and smile is not None:
            novel_l.append(True)
        else:
            novel_l.append(False)
    novels = np.array(novel_l)
    return novels


def drugcandidate_scores(mols, data):
    scores = (_constant_bump(
        water_octanol_partition_coefficient_scores(mols, norm=True), 0.210, 0.945)
              + synthetic_accessibility_score_scores(mols, norm=True)
              + novel_scores(
                mols, data) + (1 - novel_scores(mols, data)) * 0.3) / 4

    return scores


def valid_scores(mols):
    t = map(valids, mols)
    l = list(t)
    trues = []
    for mo in l:
        if mo[0] is not None:
            trues.append(True)
        else:
            trues.append(False)
    score = np.array(trues, dtype=np.float32)
    return score


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
    def novel_scores(mols, smiles):
        return np.array(
            list(map(lambda x: MolecularMetrics.valid_lambda(x) and Chem.MolToSmiles(x) not in smiles, mols)))

    @staticmethod
    def novel_filter(mols, smiles):
        return list(filter(lambda x: MolecularMetrics.valid_lambda(x) and Chem.MolToSmiles(x) not in smiles, mols))

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

    # @staticmethod
    # def diversity_scores(mols, data):
    #     rand_mols = np.random.choice(data, 100)
    #     fps = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048) for mol in rand_mols]
    #
    #     scores = np.array(
    #         list(map(lambda x: MolecularMetrics.__compute_diversity(x, fps) if x is not None else 0, mols)))
    #     scores = np.clip(MolecularMetrics.remap(scores, 0.9, 0.945), 0.0, 1.0)
    #
    #     return scores

    @staticmethod
    def synthetic_accessibility_score_scores(mols, norm=False):
        scores = [MolecularMetrics._compute_SAS(mol) if mol is not None else None for mol in mols]
        scores = np.array(list(map(lambda x: 10 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(scores, 5, 1.5), 0.0, 1.0) if norm else scores

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
