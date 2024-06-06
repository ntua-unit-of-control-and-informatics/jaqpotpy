"""
Tests for Docking
"""
import os
import platform
import unittest
import logging
#import pytest
import numpy as np
import jaqpotpy as jp
from jaqpotpy.descriptors.base_classes import ComplexFeaturizer
from jaqpotpy.models import Model
from jaqpotpy.docking.pose_generation import PoseGenerator
import rdkit.Chem
import rdkit.Chem
#from jaqpotpy.docking.utils import create_hydrated_pdbqt_pdb
# pylint: disable=no-member

IS_WINDOWS = platform.system() == 'Windows'


class TestDocking(unittest.TestCase):
  """
  Does sanity checks on pose generation.
  """

  def setUp(self):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # self.protein_file = os.path.join(current_dir, "1jld_protein.pdb")
    # self.protein_file = os.path.join(current_dir, "7zb6.pdb")
    # self.protein_file = os.path.join(current_dir, "0_a.pdb")
    self.protein_file = os.path.join(current_dir, "117_a.pdb")
    self.ligand_file = os.path.join(current_dir, "mulno.sdf")
    # self.ligand_file = os.path.join(current_dir, "1jld_ligand.sdf")

    # self.protein_file = os.path.join(current_dir, "1a9m_pocket.pdb")
    # self.ligand_file = os.path.join(current_dir, "1a9m_ligand.sdf")
    # self.ligand_file = os.path.join(current_dir, "ZINC000787318646.sdf")

  #@pytest.mark.slow
  @unittest.skip("Docking has not been integrated in the latest jaqpotpy version")
  def test_docker_init(self):
    """Test that Docker can be initialized."""
    vpg = jp.docking.VinaPoseGenerator()
    jp.docking.Docker(vpg)

  @unittest.skip("Docking has not been integrated in the latest jaqpotpy version")
  def test_tdc(self):
    from jaqpotpy.datasets import TorchGraphDataset
    from jaqpotpy.models import Evaluator
    import torch
    from jaqpotpy.models import GCN_V1, MolecularTorchGeometric, AttentiveFP

    from jaqpotpy.descriptors.molecular import AttentiveFPFeaturizer

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    #from tdc.single_pred import ADME
    #data = ADME(name='BBB_Martins')
    data = None

    split = data.get_split()
    featurizer = AttentiveFPFeaturizer()
    train_dataset = TorchGraphDataset(smiles=split['train']['Drug'], y=split['train']['Y'], task='classification'
                                , featurizer=featurizer)

    test_dataset = TorchGraphDataset(smiles=split['test']['Drug'], y=split['test']['Y'], task='classification'
                                , featurizer=featurizer)

    train_dataset.create()
    test_dataset.create()
    val = Evaluator()
    val.dataset = test_dataset
    val.register_scoring_function('Accuracy', accuracy_score)
    val.register_scoring_function('F1', f1_score)
    val.register_scoring_function('Roc Auc', roc_auc_score)

    model = AttentiveFP(in_channels=39, hidden_channels=100, out_channels=2, edge_dim=10, num_layers=6,
                        num_timesteps=2).jittable()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    m = MolecularTorchGeometric(dataset=train_dataset
                                , model_nn=model, eval=val
                                , train_batch=262, test_batch=200
                                , epochs=840, optimizer=optimizer, criterion=criterion, device="cpu").fit()

    m.eval()

    mol_m = m.create_molecular_model()
    mol_m.model_title = "BBB_MODEL"
    mol_m.save()
  
  @unittest.skip("Docking has not been integrated in the latest jaqpotpy version")
  def test_smi_file(self):
      from rdkit import Chem
      from rdkit.Chem import AllChem
      suppl = Chem.SmilesMolSupplier("/Users/pantelispanka/Downloads/bbb_v01.smi")
      for mol in suppl:
          id = mol.GetProp('id')
          smile = Chem.MolToSmiles(mol)
          mol = Chem.AddHs(mol)
          AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
          Chem.MolToMolBlock(mol)
          # ligand = "./" + str(id) + "_a_syn_ligand.sdf"
          ligand = "/Users/pantelispanka/Jaqpot/jaqpotpy/jaqpotpy/docking/test/bbb_v01_0_a_syn_ligand.sdf"
          # with Chem.SDWriter(ligand) as w:
          #     w.write(mol)
          vpg = jp.docking.VinaPoseGenerator(prepare_protein=True)
          docker = jp.docking.Docker(vpg)
          docked_outputs = docker.dock((self.protein_file, ligand),
                                       centroid=(80, 38, 48),
                                       box_dims=(12, 12, 12),
                                       exhaustiveness=12,
                                       num_modes=1,
                                       out_dir="./tmp",
                                       use_pose_generator_scores=True)

          # Check only one output since num_modes==1
          docked_outputs = list(docked_outputs)
          print(docked_outputs)
          print(len(docked_outputs))
          print(len(docked_outputs[0]))
          assert len(docked_outputs) == 1
          assert len(docked_outputs[0]) == 2

  @unittest.skip("Docking has not been integrated in the latest jaqpotpy version")
  def test_jaqpotpy(self):
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    from rdkit import Chem
    from jaqpotpy import Jaqpot
    from jaqpotpy.models import MolecularModel
    jaqpot = Jaqpot()
    jaqpot.request_key("pantelispanka@gmail.com", "kapan2")
    model = MolecularModel().load_from_jaqpot(jaqpot, "oZZfU6RQgLnmHgk88hnc")
    model.save()
    smiles_file = "/Users/pantelispanka/guacamol_v1_all.csv"
    df = pd.read_csv(smiles_file)
    for index, smile in enumerate(df[:100000]['SMILES']):
        print(index)
        print(smile)
        mol = Chem.MolFromSmiles(smile)
        try:
            model(mol)
            print(model.prediction)
            print(model.probability)
        except Exception as e:
            print(str(e))

  # @unittest.skip("skipping automated test")
  @unittest.skipIf(IS_WINDOWS, "vina is not supported in windows")
  @unittest.skip("Docking has not been integrated in the latest jaqpotpy version")
  #@pytest.mark.slow
  def test_docker_dock(self):
    """Test that Docker can dock."""
    # We provide no scoring model so the docker won't score
    vpg = jp.docking.VinaPoseGenerator(calc_charges=False, add_hydrogens=False)
    docker = jp.docking.Docker(vpg)
    docked_outputs = docker.dock((self.protein_file, self.ligand_file),
                                 exhaustiveness=12,
                                 num_modes=4,
                                 out_dir="./tmp",
                                 use_pose_generator_scores=True)

    print(docked_outputs)
    print(list(docked_outputs))
    # Check only one output since num_modes==1
    assert len(list(docked_outputs)) == 4

  # @unittest.skip("skipping automated test")
  @unittest.skipIf(IS_WINDOWS, "vina is not supported in windows")
  @unittest.skip("Docking has not been integrated in the latest jaqpotpy version")
  #@pytest.mark.slow
  def test_docker_pose_generator_scores_with_smiles(self):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import pandas as pd
    """Test that Docker can get scores from pose_generator."""
    # We provide no scoring model so the docker won't score

    smiles_file = "/Users/pantelispanka/guacamol_v1_all.csv"
    df = pd.read_csv(smiles_file)
    for index, smile in enumerate(df[:100]['SMILES']):
        print(index)
        print(smile)
        # pdb_qt = create_hydrated_pdbqt_pdb(self.protein_file)
        # smile = "C1CNCCN(C1)S(=O)(=O)C2=CC=CC3=C2C=CN=C3"
        mol = Chem.MolFromSmiles(smile)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
        Chem.MolToMolBlock(mol)
        ligand = "./" + str(index) + "_a_syn_ligand.sdf"
        with Chem.SDWriter(ligand) as w:
            w.write(mol)
        vpg = jp.docking.VinaPoseGenerator(prepare_protein=True)
        docker = jp.docking.Docker(vpg)
        docked_outputs = docker.dock((self.protein_file, ligand),
                                     centroid=(80, 38, 48),
                                     box_dims=(12, 12, 12),
                                     exhaustiveness=12,
                                     num_modes=1,
                                     out_dir="./tmp",
                                     use_pose_generator_scores=True)

        # Check only one output since num_modes==1
        docked_outputs = list(docked_outputs)
        print(docked_outputs)
        print(len(docked_outputs))
        print(len(docked_outputs[0]))
        assert len(docked_outputs) == 1
        assert len(docked_outputs[0]) == 2


  # @unittest.skip("skipping automated test")
  @unittest.skipIf(IS_WINDOWS, "vina is not supported in windows")
  @unittest.skip("Docking has not been integrated in the latest jaqpotpy version")
  #@pytest.mark.slow
  def test_docker_pose_generator_scores_with_pdbqt(self):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import pandas as pd
    """Test that Docker can get scores from pose_generator."""
    # We provide no scoring model so the docker won't score

    # pdb_qt = create_hydrated_pdbqt_pdb(self.protein_file)
    smile = "C1CNCCN(C1)S(=O)(=O)C2=CC=CC3=C2C=CN=C3"
    mol = Chem.MolFromSmiles(smile)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
    Chem.MolToMolBlock(mol)
    ligand = "./fasudil.sdf"
    with Chem.SDWriter(ligand) as w:
        w.write(mol)
    vpg = jp.docking.VinaPoseGenerator(prepare_protein=True)
    docker = jp.docking.Docker(vpg)
    docked_outputs = docker.dock((self.protein_file, ligand),
                                 centroid=(80, 38, 48),
                                 box_dims=(21, 21, 29),
                                 exhaustiveness=18,
                                 num_modes=1,
                                 out_dir="./tmp",
                                 use_pose_generator_scores=True)

    # Check only one output since num_modes==1
    docked_outputs = list(docked_outputs)
    print(docked_outputs)
    print(len(docked_outputs))
    print(len(docked_outputs[0]))
    assert len(docked_outputs) == 1
    assert len(docked_outputs[0]) == 2





  # @unittest.skip("skipping automated test")
  @unittest.skipIf(IS_WINDOWS, "vina is not supported in windows")
  @unittest.skip("Docking has not been integrated in the latest jaqpotpy version")

  #@pytest.mark.slow
  def test_docker_pose_generator_scores(self):
    """Test that Docker can get scores from pose_generator."""
    # We provide no scoring model so the docker won't score
    vpg = jp.docking.VinaPoseGenerator()
    docker = jp.docking.Docker(vpg)
    docked_outputs = docker.dock((self.protein_file, self.ligand_file),
                                 exhaustiveness=4,
                                 num_modes=4,
                                 out_dir="./tmp",
                                 use_pose_generator_scores=True)

    # Check only one output since num_modes==1
    docked_outputs = list(docked_outputs)
    print(docked_outputs)
    print(len(docked_outputs))
    print(len(docked_outputs[0]))
    assert len(docked_outputs) == 4
    assert len(docked_outputs[0]) == 2

  @unittest.skipIf(IS_WINDOWS, "vina is not supported in windows")
  @unittest.skip("skipping automated test")
  #@pytest.mark.slow
  def test_docker_pose_generator_scores_on_pocket(self):
    """Test that Docker can get scores from pose_generator."""
    # We provide no scoring model so the docker won't score
    vpg = jp.docking.VinaPoseGenerator()
    docker = jp.docking.Docker(vpg)
    docked_outputs = docker.dock((self.protein_file, self.ligand_file),
                                 exhaustiveness=40,
                                 num_modes=4,
                                 out_dir="./tmp",
                                 use_pose_generator_scores=True)

    # Check only one output since num_modes==1
    docked_outputs = list(docked_outputs)
    print(docked_outputs)
    assert len(docked_outputs) == 1
    assert len(docked_outputs[0]) == 2

  @unittest.skipIf(IS_WINDOWS, "vina is not supported in windows")
  @unittest.skip("skipping automated test")
  #@pytest.mark.slow
  def test_docker_specified_pocket(self):
    """Test that Docker can dock into spec. pocket."""
    # Let's turn on logging since this test will run for a while
    logging.basicConfig(level=logging.INFO)
    vpg = jp.docking.VinaPoseGenerator()
    docker = jp.docking.Docker(vpg)
    docked_outputs = docker.dock((self.protein_file, self.ligand_file),
                                 centroid=(10, 10, 10),
                                 box_dims=(10, 10, 10),
                                 exhaustiveness=1,
                                 num_modes=1,
                                 out_dir="./tmp")

    # Check returned files exist
    assert len(list(docked_outputs)) == 1

  @unittest.skipIf(IS_WINDOWS, "vina is not supported in windows")
  @unittest.skip("Docking has not been integrated in the latest jaqpotpy version")
  #@pytest.mark.slow
  def test_pocket_docker_dock(self):
    """Test that Docker can find pockets and dock dock."""
    # Let's turn on logging since this test will run for a while
    logging.basicConfig(level=logging.INFO)
    pocket_finder = jp.docking.ConvexHullPocketFinder()
    vpg = jp.docking.VinaPoseGenerator(pocket_finder=pocket_finder)
    docker = jp.docking.Docker(vpg)
    docked_outputs = docker.dock((self.protein_file, self.ligand_file),
                                 exhaustiveness=1,
                                 num_modes=1,
                                 num_pockets=1,
                                 out_dir="./tmp")

    # Check returned files exist
    assert len(list(docked_outputs)) == 1

  #@pytest.mark.slow
  @unittest.skip("Docking has not been integrated in the latest jaqpotpy version")
  def test_scoring_model_and_featurizer(self):
    """Test that scoring model and featurizer are invoked correctly."""

    class DummyFeaturizer(ComplexFeaturizer):

      def featurize(self, complexes, *args, **kwargs):
        return np.zeros((len(complexes), 5))

    class DummyModel(Model):

      def predict(self, dataset, *args, **kwargs):
        return np.zeros(len(dataset))

    class DummyPoseGenerator(PoseGenerator):

      def generate_poses(self, *args, **kwargs):
        return [None]

    featurizer = DummyFeaturizer()
    scoring_model = DummyModel()
    pose_generator = DummyPoseGenerator()
    docker = jp.docking.Docker(pose_generator, featurizer, scoring_model)
    outputs = docker.dock(None)
    assert list(outputs) == [(None, np.array([0.]))]