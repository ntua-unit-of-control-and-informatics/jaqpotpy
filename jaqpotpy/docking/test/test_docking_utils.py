"""
Tests for Docking
"""
import os
import platform
import unittest
#import pytest
import logging
import numpy as np
import jaqpotpy as jp
from jaqpotpy.descriptors.base_classes import ComplexFeaturizer
from jaqpotpy.models import Model
from jaqpotpy.docking.pose_generation import PoseGenerator

IS_WINDOWS = platform.system() == 'Windows'


class TestDockingUtils(unittest.TestCase):
  """
  Does sanity checks on pose generation.
  """

  def setUp(self):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # self.protein_file = os.path.join(current_dir, "1jld_protein.pdb")
    # self.protein_file = os.path.join(current_dir, "7zb6.pdb")
    # self.ligand_file = os.path.join(current_dir, "mulno.sdf")
    # self.ligand_file = os.path.join(current_dir, "1jld_ligand.sdf")

    self.protein_file = os.path.join(current_dir, "1a9m_pocket.pdb")
    self.ligand_file = os.path.join(current_dir, "1a9m_ligand.sdf")


  #@pytest.mark.slow
  @unittest.skip("Docking has not been integrated in the latest jaqpotpy version")
  def test_docker_init(self):
    """Test that Docker can be initialized."""
    vpg = jp.docking.VinaPoseGenerator()
    jp.docking.Docker(vpg)

  @unittest.skip("test only for conda env")
  def test_docking_utils(self):
      from jaqpotpy.docking.utils import create_hydrated_pdbqt_pdb
      protein_pdbqt = create_hydrated_pdbqt_pdb(self.protein_file, "./tmp")
      print(protein_pdbqt)


  @unittest.skipIf(IS_WINDOWS, "vina is not supported in windows")
  @unittest.skip("Docking has not been integrated in the latest jaqpotpy version")
  #@pytest.mark.slow
  def test_docker_dock(self):
    """Test that Docker can dock."""
    # We provide no scoring model so the docker won't score
    vpg = jp.docking.VinaPoseGenerator()
    docker = jp.docking.Docker(vpg)
    docked_outputs = docker.dock((self.protein_file, self.ligand_file),
                                 exhaustiveness=10,
                                 num_modes=10,
                                 out_dir="./tmp")

    print(docked_outputs)
    # Check only one output since num_modes==1
    assert len(list(docked_outputs)) == 10
