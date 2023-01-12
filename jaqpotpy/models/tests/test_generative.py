import unittest
from jaqpotpy.datasets.molecular_datasets import SmilesDataset
from jaqpotpy.descriptors.molecular import MolGanFeaturizer, GraphMatrix
from jaqpotpy.models.evaluator import GenerativeEvaluator
from jaqpotpy.models.generative.models import GanMoleculeGenerator\
    , GanMoleculeDiscriminator, MoleculeDiscriminator
from jaqpotpy.models.generative.gan import GanSolver
from torch.utils.data import DataLoader
import torch.nn.functional as F
from rdkit import Chem
import torch

from jaqpotpy.models.generative.molecular_metrics import diversity_scores \
    , drugcandidate_scores, synthetic_accessibility_score_scores, valid_mean \
    , quantitative_estimation_druglikeness_scores, novel_score \
    , water_octanol_partition_coefficient_scores, unique_total_score, valid_scores


class TestModels(unittest.TestCase):


    def setUp(self):
        from tdc.generation import MolGen
        try:
            self.smiles = []
            with open("./data/zinc.tab", "r") as f:
                for l in f.readlines():
                    if l != "smiles\n":
                        self.smiles.append(l.replace('\n', "").replace("\"", ""))
        except FileNotFoundError:
            self.data = MolGen(name='ZINC')
            self.smiles = self.data.__dict__['smiles_lst']

        self.mols_test = [
            "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1",
            "C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1",
            "N#Cc1ccc(-c2ccc(O[C@@H](C(=O)N3CCCC3)c3ccccc3)cc2)cc1",
            "CCOC(=O)[C@@H]1CCCN(C(=O)c2nc(-c3ccc(C)cc3)n3c2CCCCC3)C1",
            "N#CC1=C(SCC(=O)Nc2cccc(Cl)c2)N=C([O-])[C@H](C#N)C12CCCCC2",
            "CC[NH+](CC)[C@](C)(CC)[C@H](O)c1cscc1Br",
            "COc1ccc(C(=O)N(C)[C@@H](C)C/C(N)=N/O)cc1O",
            "O=C(Nc1nc[nH]n1)c1cccnc1Nc1cccc(F)c1",
            "Cc1c(/C=N/c2cc(Br)ccn2)c(O)n2c(nc3ccccc32)c1C#N",
            "C[C@@H]1CN(C(=O)c2cc(Br)cn2C)CC[C@H]1[NH3+]",
            "CCOc1ccc(OCC)c([C@H]2C(C#N)=C(N)N(c3ccccc3C(F)(F)F)C3=C2C(=O)CCC3)c1",
            "Cc1ccc2nc(S[C@H](C)C(=O)NC3CCC(C)CC3)n(C)c(=O)c2c1",
            "O=C(N1CCc2c(F)ccc(F)c2C1)C1(O)Cc2ccccc2C1",
            "Cc1ccccc1C(=O)N1CCC2(CC1)C[C@H](c1ccccc1)C(=O)N2C",
            "CCCc1cc(NC(=O)CN2C(=O)NC3(CCC(C)CC3)C2=O)n(C)n1",
            "CC(C)Cc1nc(SCC(=O)NC[C@@H]2CCCO2)c2c(=O)n(C)c(=O)n(C)c2n1",
            "Cc1ccc(CNC(=O)c2ccccc2NC(=O)[C@@H]2CC(=O)N(c3ccc(C)cc3)C2)cc1",
            "CCCCC(=O)NC(=S)Nc1ccccc1C(=O)N1CCOCC1",
            "Cc1c(NC(=O)CSc2nc3sc4c(c3c(=O)[nH]2)CCCC4)c(=O)n(-c2ccccc2)n1C",
            "CC(C)[C@@H](Oc1cccc(Cl)c1)C(=O)N1CCC(n2cccn2)CC1",
            "CCN(CC)C(=O)C[C@@H](C)[NH2+][C@H](C)c1cccc(F)c1F",
            "Cc1nc2c(c(Nc3ncc(C)s3)n1)CCN(C(=O)CCc1ccccc1)C2",
            "O=C(NCCNC(=O)N1C[C@H]2CC=CC[C@@H]2C1)c1cccnc1",
            "O=c1n(CCO)c2ccccc2n1CCO",
            "COC(=O)Cc1csc(NC(=O)Cc2coc3cc(C)ccc23)n1",
            "Cc1ccc(N2CC[C@@H](NS(=O)(=O)c3ccccc3C)C2=O)cc1C",
            "CC[C@H](C)C[C@@H](C)NC(=O)N1CCN(CC(=O)NC2CC2)CC1",
            "CC(=O)Nc1c2n(c3ccccc13)C[C@](C)(C(=O)NC1CCCCC1)N(C1CCCCC1)C2=O",
            "N#Cc1ccncc1NC[C@@H]1C[C@@]12CCc1ccccc12",
            "Cc1cccn2c(=O)c(C(=O)NC[C@H]3CCO[C@@H]3C(C)C)cnc12",
            "CNC(=O)c1ccc(/C=C/C(=O)Nc2c(C)cc(C)nc2Cl)cc1",
            "CC1=C(CNC(=O)c2cc(-c3ccccc3)nc3c2CNN3C(C)C)CN=N1",
            "C[C@@H](NC(=O)COC(=O)/C=C/c1ccc(Cl)cc1)c1ccccc1",
            "CCc1ccc(N(Cc2ccc(C)s2)C(=O)c2ccc(=O)n(C)n2)cc1",
            "CCOC(=O)c1nnc2ccccc2c1N1CC[C@@H]([NH+](CC)CC)C1",
            "Cc1ccc(C#N)cc1S(=O)(=O)NCc1ccnc(OC(C)(C)C)c1",
            "O=C(O[C@H]1CCOC1)C1(c2ccc(Cl)c(Cl)c2)CCC1",
            "CCC[NH2+][C@@H]1COC[C@H]1C(=O)NCc1cscc1C",
            "O=C(NCc1nccc2ccccc12)c1ccc[nH]c1=O",
            "CC(=O)c1ccc(S(=O)(=O)N2CCCC[C@H]2C)cc1",
        ]


    def test_smiles_parser(self):
        for s in self.smiles[0:10080]:
            print(s)
            mol = Chem.MolFromSmiles(s)
            feat = MolGanFeaturizer(max_atom_count=42)
            f = feat.featurize(s)
            print(f)
            print(mol)

    def test_qm9(self):

        mols = Chem.SDMolSupplier('./data/gdb9.sdf')
        smiles_dif = []
        for m in mols:
            try:
                sm = Chem.MolToSmiles(m)
                print("SMILES FOR FEAT: " + Chem.MolToSmiles(m))
                featurizer = MolGanFeaturizer(max_atom_count=12, kekulize=True, sanitize=False)
                mf = featurizer.featurize(Chem.MolToSmiles(m))
                mfde = featurizer.defeaturize(mf)

                if mfde:
                    sm_c = Chem.MolToSmiles(mfde[0])
                    if sm != sm_c:
                        smiles_dif.append(sm)
                    print("SMILES DEFEAT: " + Chem.MolToSmiles(mfde[0]))
                print(smiles_dif)
            except Exception as e:
                pass

    def test_two_graphs(self):
        featurizer = MolGanFeaturizer(max_atom_count=12, kekulize=True, sanitize=False)
        smile_1 = "CC(=O)OCN"
        print("Featurizing smile 1: " + smile_1)
        g = featurizer.featurize(smile_1)
        g_def = featurizer.defeaturize(g)
        smile_2 = Chem.MolToSmiles(g_def[0])
        print("Deafeturized smile 1 : " + smile_2)

        smile_3 = "C[NH+]1CC(O)C1"
        g_2 = featurizer.featurize(smile_3)
        print("COMPARING GRAPHS")
        self.assertEqual(g, g)

    def test_smiles_list(self):
        featurizer = MolGanFeaturizer(max_atom_count=12, kekulize=True, sanitize=False)
        smiles = ['[CH]C#C[CH]', '[CH]C#C[NH3+]', 'C[NH+]1CC1', 'COC(C)[NH-]', '[NH-]C1CCO1', 'CC1C[NH+]1C', '[NH-][CH+]OC=O',
         'C[NH2+][CH+]OC', 'CC[NH+]1CC1', 'C#C/C(C)=N/O', 'C/C(CO)=N\\O', 'CC/C(C)=N/O', '[NH-]c1[nH+]cco1',
         '[NH-]C1OC=CO1', 'CC([NH-])OC=O', 'CC(=O)O[CH+][NH-]', '[NH-][CH+]OC(N)=O', 'C[NH+](C)CC#N', 'C[NH+](C)CC=O',
         '[NH3+]CCC(=O)[O-]', 'NC(=[NH2+])C(=O)[O-]', 'C[NH+]1CC1(C)C', 'CC1CC([NH-])O1', 'C[NH+]1CC(=O)C1',
         'C[NH+]1CC(O)C1', 'CC12C[NH+](C1)C2', 'OC12C[NH+](C1)C2', 'C/N=C(\\N)C#N', 'C/N=C(\\N)C=O', 'C[NH2+]C(C)OC']
        for s in smiles:
            print("Featurizing smile 1: " + s)
            g = featurizer.featurize(s)
            g_def = featurizer.defeaturize(g)
            smile_2 = Chem.MolToSmiles(g_def[0])
            print("Deafeturized smile 1 : " + smile_2)

            smile_3 = "C[NH+]1CC(O)C1"
            g_2 = featurizer.featurize(smile_3)
            print("COMPARING GRAPHS")
            self.assertEqual(g, g)

    def test_dataset(self, max_atoms=40, length=100):
        featurizer = MolGanFeaturizer(max_atom_count=max_atoms, kekulize=True, sanitize=True)
        self.dataset = SmilesDataset(smiles=self.smiles[0:length], task="generation", featurizer=featurizer)
        self.dataset.create()
        # for s in self.smiles:
        #     from jaqpotpy.cfg import config
        #     config.verbose = False
        #     print("------")
        #     mol = Chem.MolFromSmiles(s)
        #     # mol = Chem.Kekulize(mol)
        #     print(s)
        #     item = featurizer.featurize(s)
        #     mol = featurizer.defeaturize(item)
        #     try:
        #         smiles = Chem.MolToSmiles(mol[0])
        #     except Exception as e:
        #         continue
        #         print(e)
        # item = featurizer.featurize(self.smiles[0])
        # print(item[0].node_features.shape)
        # print(item[0].node_features)
        # print(item[0].adjacency_matrix.shape)
        # print(item[0].adjacency_matrix)
        # mol = featurizer.defeaturize(item)
        # print(Chem.MolToSmiles(mol[0]))
        # for i in range(len(self.dataset.smiles)):
        #
        #     ditem = self.dataset.__getitem__(i)
        #     print(self.dataset.smiles[i])
        #     g1 = GraphMatrix(ditem[2][1].cpu().detach().numpy(), ditem[2][0].cpu().detach().numpy())
        #     t = featurizer.defeaturize(g1)
        #     print(Chem.MolToSmiles(t[0]))
        #
        #
        # data_loader = DataLoader(dataset=self.dataset, **{'batch_size': 4, 'shuffle': False, 'num_workers': 0})
        #
        # items = self.dataset.smiles
        # print(items)
        # print(len(self.dataset.smiles))
        # for ditems in data_loader:
        #     nod_fs = torch.split(ditems[2][0], 1)
        #     adc_mat = torch.split(ditems[2][1], 1)
        #     for ind, n in enumerate(nod_fs):
        #         g2 = GraphMatrix(torch.squeeze(adc_mat[ind]).cpu().detach().numpy(), torch.squeeze(n).cpu().detach().numpy())
        #         t = featurizer.defeaturize(g2)
                # print(Chem.MolToSmiles(t[0]))

    def test_defeat(self):
        # smiles = "N.[CH2-]c1cc(O)co1"
        smiles = "C=C1C=CC(=O)O1.N"
        mol = Chem.MolFromSmiles(smiles)
        print(smiles)
        feat = MolGanFeaturizer(max_atom_count=42, kekulize=True, sanitize=True)
        f = feat.featurize(smiles)
        print(f)
        mol = feat.defeaturize(f)
        print(Chem.MolToSmiles(mol))

    def test_defeat_two(self):
        print(self.smiles[1])
        mol = Chem.MolFromSmiles(self.smiles[1])
        feat = MolGanFeaturizer(max_atom_count=42)
        f = feat.featurize(mol)
        print(f)
        mol = feat.defeaturize(f)
        print(Chem.MolToSmiles(mol))

    def test_generator_model(self):
        max_atoms = 40
        self.test_dataset(max_atoms=max_atoms)
        generator = GanMoleculeGenerator([128, 256, 524], 8, max_atoms, 5, max_atoms, 0.5)
        print(generator.eval())
        data_loader = DataLoader(dataset=self.dataset, **{'batch_size': 1, 'shuffle': False, 'num_workers': 0})
        for i in data_loader:
            # print(i)
            samples = generator.sample_generator(data_loader.batch_size)
            # print(samples)

            out1, out2 = generator(samples)
            g = GraphMatrix(out1.cpu().detach().numpy(), out2.cpu().detach().numpy())
            self.dataset.featurizer.defeaturize(g)
            print(out1.size())
            print(out2.size())
            edges_hat, nodes_hat = self.postprocess((out1, out2), 'soft_gumbel')
            # print(edges_hat.size())
            # print(edges_hat)
            # print(nodes_hat.size())

    def test_discriminator(self):
        import torch
        max_atoms = 42
        atoms_encoded = 12
        self.test_dataset(max_atoms=max_atoms)
        generator = GanMoleculeGenerator([128, 256, 524], 8, max_atoms, 5, atoms_encoded, 0.5)
        discriminator = MoleculeDiscriminator([[128, 64], 128, [128, 64]], atoms_encoded, 5 - 1, 0.5)
        data_loader = DataLoader(dataset=self.dataset, **{'batch_size': 4, 'shuffle': False, 'num_workers': 0})
        for i in data_loader:
            j = i[0]
            # print(i[0])
            samples = generator.sample_generator(data_loader.batch_size)
            # print(samples)
            out1, out2 = generator(samples)

            print(out1.size())
            print(out2.size())

            edges_hard, nodes_hard = torch.max(out1, -1)[1], torch.max(out2, -1)[1]

            out_d = discriminator(out1, None, out2)
            print(out_d)
            p = self.postprocess((out1, out2), "hard_gumbel")

            print(p[0].shape)
            print(p[1].shape)
            g = GraphMatrix(torch.squeeze(p[0]).cpu().detach().numpy(), torch.squeeze(p[1]).cpu().detach().numpy())
            mol = self.dataset.featurizer.defeaturize(g)
            print(mol)

            nod_fs = torch.split(i[2][0], 1)
            adc_mat = torch.split(i[2][1], 1)
            for ind, n in enumerate(nod_fs):
                g2 = GraphMatrix(torch.squeeze(adc_mat[ind]).cpu().detach().numpy(), torch.squeeze(n).cpu().detach().numpy())
                t = self.dataset.featurizer.defeaturize(g2)
                print(Chem.MolToSmiles(t[0]))

    def test_generator_optim(self):
        import numpy as np
        max_atoms = 42
        atoms_encoded = 12
        generator = GanMoleculeGenerator([128, 256, 524], 8, max_atoms, 5, atoms_encoded, 0.5)
        g_optim = torch.optim.Adam(generator.parameters(), lr=0.1, weight_decay=0.01)
        loss = torch.from_numpy(np.full(3, 0.3))
        loss = torch.mean(loss)
        loss.requires_grad = True
        g_optim.zero_grad()
        loss.backward()
        g_optim.step()

    def test_gan_solver(self):
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.info')

        from jaqpotpy.cfg import config
        smiles_len = 1000
        config.verbose = False
        max_atoms = 42
        atoms_encoded = 12
        self.test_dataset(max_atoms=max_atoms, length=smiles_len)
        generator = GanMoleculeGenerator([128, 256, 524], 8, max_atoms, 5, atoms_encoded, 0.5)
        discriminator = MoleculeDiscriminator([[128, 64], 128, [128, 64]], atoms_encoded, 5 - 1, 0.5)
        solver = GanSolver(generator=generator
                           , discriminator=discriminator, dataset=self.dataset
                           , la=0.1, g_lr=0.1, d_lr=0.1, batch_size=42, val_at=20, epochs=30)
        solver.fit()

    def test_with_qm_9(self):
        mols = Chem.SDMolSupplier('./data/gdb9.sdf')
        max_atoms = 10
        atoms_encoded = 6
        smiles_gen = []
        for m in mols:
            try:
                smile = Chem.MolToSmiles(m)
                smiles_gen.append(smile)
            except Exception as e:
                continue
        featurizer = MolGanFeaturizer(max_atom_count=10, kekulize=True, sanitize=True, atom_labels=[0, 6, 7, 8, 9, 5])
        self.dataset = SmilesDataset(smiles=smiles_gen[1000:5000], task="generation", featurizer=featurizer)
        self.dataset.create()
        generator = GanMoleculeGenerator([128, 256, 524], 8, max_atoms, 5, atoms_encoded, 0.5)
        discriminator = MoleculeDiscriminator([[128, 64], 128, [128, 64]], atoms_encoded, 5 - 1, 0.5)
        solver = GanSolver(generator=generator
                           , discriminator=discriminator, dataset=self.dataset
                           , la=1, g_lr=0.1, d_lr=0.1, batch_size=42, val_at=20, epochs=30)
        solver.fit()

    def test_with_evaluator(self):
        mols = Chem.SDMolSupplier('./data/gdb9.sdf')
        max_atoms = 9
        atoms_encoded = 5
        featurizer = MolGanFeaturizer(max_atom_count=9, kekulize=False, sanitize=True, atom_labels=[0, 6, 7, 8, 9])
        smiles_gen = []
        from jaqpotpy.cfg import config
        config.verbose = False
        i = 0
        for m in mols:
            i += 1
            if i > 30000:
                break
            try:
                smile = Chem.MolToSmiles(m)
                feat = featurizer.featurize(smile)
                i += 1
                if feat[0]:
                    smiles_gen.append(smile)
            except Exception as e:
                continue
        config.verbose = True
        smiles_train = []
        for st in smiles_gen[1000:10000]:
            smiles_train.append(st)
        smiles_test = []
        for stt in smiles_gen[12000:13969]:
            smiles_test.append(stt)

        self.dataset = SmilesDataset(smiles=smiles_train, task="generation", featurizer=featurizer)
        self.dataset.create()
        generator = GanMoleculeGenerator([128, 256, 524], 8, max_atoms, 5, atoms_encoded, 0.5)
        discriminator = MoleculeDiscriminator([[128, 64], 128, [128, 64]], atoms_encoded, 5 - 1, 0.5)

        gen_eval = GenerativeEvaluator()
        # gen_eval.register_scoring_function("Valid all", valid_mean)
        gen_eval.register_scoring_function("Valid", valid_scores)
        gen_eval.register_scoring_function("QED", quantitative_estimation_druglikeness_scores)
        # gen_eval.register_scoring_function("BBB", bbb_function)
        # gen_eval.register_scoring_function("Synthetic Accessibility", synthetic_accessibility_score_scores)
        # gen_eval.register_scoring_function("Novel", novel_score)
        # gen_eval.register_scoring_function("Unique", unique_total_score)
        # gen_eval.register_scoring_function("Diversity", diversity_scores)
        # gen_eval.register_scoring_function("Water Oct", water_octanol_partition_coefficient_scores)
        # gen_eval.register_scoring_function("Drugcandidate Scores", drugcandidate_scores)

        # gen_eval.register_scoring_function("Novel all", novel_score)
        gen_eval.register_dataset(smiles_test)
        # gen_eval.register_evaluation_function("BBB Mean", bbb_mean_function)
        gen_eval.register_evaluation_function("Valid all", valid_mean)
        gen_eval.register_evaluation_function("Synthetic Accessibility", synthetic_accessibility_score_scores)
        gen_eval.register_evaluation_function("Novel", novel_score)
        gen_eval.register_evaluation_function("Unique", unique_total_score)
        gen_eval.register_evaluation_function("Water Oct", water_octanol_partition_coefficient_scores)
        gen_eval.register_evaluation_function("Drugcandidate Scores", drugcandidate_scores)

        solver = GanSolver(generator=generator
                           , discriminator=discriminator, evaluator=gen_eval, dataset=self.dataset
                           , la=0, g_lr=0.0001, d_lr=0.0001, lamda_gradient_penalty=10.0
                           , batch_size=42, val_at=1, epochs=30)
        solver.fit()


    def test_with_evaluator_big(self):
        smiles = []
        try:
            with open("./data/zinc.tab", "r") as f:
                for l in f.readlines():
                    if l != "smiles\n":
                        smiles.append(l.replace('\n', "").replace("\"", ""))
        except Exception as e:
            print(e)

        max_atoms = 20
        atoms_encoded = 12
        featurizer = MolGanFeaturizer(max_atom_count=max_atoms, kekulize=False, sanitize=True, atom_labels=[0, 6, 7, 8, 9, 5, 53, 1, 16, 15, 17, 35] )
        smiles_gen = []
        from jaqpotpy.cfg import config
        config.verbose = False
        for m in smiles[0:60000]:
            feat = featurizer.featurize(m)
            if feat[0]:
                smiles_gen.append(m)
        config.verbose = True
        smiles_train = []
        for st in smiles_gen[0:4000]:
            smiles_train.append(st)
        smiles_test = []
        for stt in smiles_gen[4000:4500]:
            smiles_test.append(stt)

        self.dataset = SmilesDataset(smiles=smiles_train, task="generation", featurizer=featurizer)
        self.dataset.create()
        generator = GanMoleculeGenerator([128, 256, 524], 8, max_atoms, 5, atoms_encoded, 0.5)
        discriminator = MoleculeDiscriminator([[128, 64], 128, [128, 64]], atoms_encoded, 5 - 1, 0.5)

        gen_eval = GenerativeEvaluator()
        # gen_eval.register_scoring_function("Valid all", valid_mean)
        gen_eval.register_scoring_function("Valid", valid_scores)
        gen_eval.register_scoring_function("QED", quantitative_estimation_druglikeness_scores)
        # gen_eval.register_scoring_function("BBB", bbb_function)
        # gen_eval.register_scoring_function("Synthetic Accessibility", synthetic_accessibility_score_scores)
        # gen_eval.register_scoring_function("Novel", novel_score)
        # gen_eval.register_scoring_function("Unique", unique_total_score)
        # gen_eval.register_scoring_function("Diversity", diversity_scores)
        # gen_eval.register_scoring_function("Water Oct", water_octanol_partition_coefficient_scores)
        # gen_eval.register_scoring_function("Drugcandidate Scores", drugcandidate_scores)

        # gen_eval.register_scoring_function("Novel all", novel_score)
        gen_eval.register_dataset(smiles_test)
        # gen_eval.register_evaluation_function("BBB Mean", bbb_mean_function)
        gen_eval.register_evaluation_function("Valid all", valid_mean)
        gen_eval.register_evaluation_function("Synthetic Accessibility", synthetic_accessibility_score_scores)
        gen_eval.register_evaluation_function("Novel", novel_score)
        gen_eval.register_evaluation_function("Unique", unique_total_score)
        gen_eval.register_evaluation_function("Water Oct", water_octanol_partition_coefficient_scores)
        gen_eval.register_evaluation_function("Drugcandidate Scores", drugcandidate_scores)

        solver = GanSolver(generator=generator
                           , discriminator=discriminator, evaluator=gen_eval, dataset=self.dataset
                           , la=0.8, g_lr=0.0001, d_lr=0.0001, lamda_gradient_penalty=10.0
                           , batch_size=42, val_at=1, epochs=130)
        solver.fit()


    def test_jaqpot_model(self):
        from jaqpotpy import Jaqpot
        from jaqpotpy.models import MolecularModel
        from jaqpotpy.cfg import config
        config.verbose = False
        jaqpot = Jaqpot()
        jaqpot.set_api_key("eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJ3Ujh3X1lGOWpKWFRWQ2x2VHF1RkswZkctQXROQUJsb3FBd0N4MmlTTWQ4In0.eyJleHAiOjE2NzI4MjE4MDQsImlhdCI6MTY3MjY0OTAwNCwiYXV0aF90aW1lIjoxNjcyNjQ5MDAzLCJqdGkiOiIxYTBiYmNjMi0wM2IzLTQ4NDYtYTg1ZS03ODcwZTdlNWE1MGYiLCJpc3MiOiJodHRwczovL2xvZ2luLmphcXBvdC5vcmcvYXV0aC9yZWFsbXMvamFxcG90IiwiYXVkIjoiYWNjb3VudCIsInN1YiI6IjI0MjVkNzYwLTAxOGQtNDA4YS1hZTBiLWNkZTRjNTYzNTRiOSIsInR5cCI6IkJlYXJlciIsImF6cCI6ImphcXBvdC11aS1jb2RlIiwibm9uY2UiOiI1MzUxZDVkZDEyNDRhNGNmYjYzZWM4OTg2OGYyYjgyM2JkWDZDbkJUViIsInNlc3Npb25fc3RhdGUiOiJhMTdhMzIxNy1jZjBmLTQxNmEtODJkMy04ZDM0OWRmNmQxZWMiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbIicqJyIsIioiXSwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6Im9wZW5pZCBqYXFwb3QtYWNjb3VudHMgZW1haWwgcHJvZmlsZSB3cml0ZSByZWFkIiwic2lkIjoiYTE3YTMyMTctY2YwZi00MTZhLTgyZDMtOGQzNDlkZjZkMWVjIiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJuYW1lIjoiUGFudGVsaXMgS2FyYXR6YXMiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJwYW50ZWxpc3BhbmthIiwiZ2l2ZW5fbmFtZSI6IlBhbnRlbGlzIiwiZmFtaWx5X25hbWUiOiJLYXJhdHphcyIsImVtYWlsIjoicGFudGVsaXNwYW5rYUBnbWFpbC5jb20ifQ.ivsqqMMMUu7fYhB-kgjmMFZKHc2q9XsX02EkKFNYFrvIX4K9EsyQKGDjudZeYU9JwR4fqNXttVOkTvsDR6g9-Hmfg5W2RdkHh3L1OhAs8U4PZ39llYGpXVv_vF7UzCH8h5EmlpBe_WBH7_HSE9pNBqtR_B8KvwCEeEllWlF8XqYdz9dezllGnGjqFldxtATk71VcDpneVx1KR2wWcj0iz1q4wLlePimE-UJw8vDn2uKy43km5LiyrAvz4RsyaGdI5lX66k7Pg0klO2rqT-xNzCwVuRv6KnESH0TIKDKE4vv9hjaEUwTPks4NjG59N-muuHebSbfPK7nDbABYfaB7yw")
        model = MolecularModel().load_from_jaqpot(jaqpot=jaqpot, id="BKsEYKTVRSKZyCjEWBzp")
        mols = Chem.SDMolSupplier('./data/gdb9.sdf')
        for mol in mols:
            # smile = Chem.MolToSmiles(mol)
            try:
                model(mol)
                print(model.prediction)
                print(model.probability)
            except Exception as e:
                continue
        print(Chem.MolToSmiles("COC(=N)NCC#N"))
        print(type('COC(=N)NCC#N').__name__)
        model("C#CC12CC(C)C1C2")
        print(model.prediction)
        print(model.probability)


from jaqpotpy import Jaqpot
from jaqpotpy.models import MolecularModel

jaqpot = Jaqpot()
jaqpot.set_api_key(
    "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJ3Ujh3X1lGOWpKWFRWQ2x2VHF1RkswZkctQXROQUJsb3FBd0N4MmlTTWQ4In0.eyJleHAiOjE2NzI4MjE4MDQsImlhdCI6MTY3MjY0OTAwNCwiYXV0aF90aW1lIjoxNjcyNjQ5MDAzLCJqdGkiOiIxYTBiYmNjMi0wM2IzLTQ4NDYtYTg1ZS03ODcwZTdlNWE1MGYiLCJpc3MiOiJodHRwczovL2xvZ2luLmphcXBvdC5vcmcvYXV0aC9yZWFsbXMvamFxcG90IiwiYXVkIjoiYWNjb3VudCIsInN1YiI6IjI0MjVkNzYwLTAxOGQtNDA4YS1hZTBiLWNkZTRjNTYzNTRiOSIsInR5cCI6IkJlYXJlciIsImF6cCI6ImphcXBvdC11aS1jb2RlIiwibm9uY2UiOiI1MzUxZDVkZDEyNDRhNGNmYjYzZWM4OTg2OGYyYjgyM2JkWDZDbkJUViIsInNlc3Npb25fc3RhdGUiOiJhMTdhMzIxNy1jZjBmLTQxNmEtODJkMy04ZDM0OWRmNmQxZWMiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbIicqJyIsIioiXSwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6Im9wZW5pZCBqYXFwb3QtYWNjb3VudHMgZW1haWwgcHJvZmlsZSB3cml0ZSByZWFkIiwic2lkIjoiYTE3YTMyMTctY2YwZi00MTZhLTgyZDMtOGQzNDlkZjZkMWVjIiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJuYW1lIjoiUGFudGVsaXMgS2FyYXR6YXMiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJwYW50ZWxpc3BhbmthIiwiZ2l2ZW5fbmFtZSI6IlBhbnRlbGlzIiwiZmFtaWx5X25hbWUiOiJLYXJhdHphcyIsImVtYWlsIjoicGFudGVsaXNwYW5rYUBnbWFpbC5jb20ifQ.ivsqqMMMUu7fYhB-kgjmMFZKHc2q9XsX02EkKFNYFrvIX4K9EsyQKGDjudZeYU9JwR4fqNXttVOkTvsDR6g9-Hmfg5W2RdkHh3L1OhAs8U4PZ39llYGpXVv_vF7UzCH8h5EmlpBe_WBH7_HSE9pNBqtR_B8KvwCEeEllWlF8XqYdz9dezllGnGjqFldxtATk71VcDpneVx1KR2wWcj0iz1q4wLlePimE-UJw8vDn2uKy43km5LiyrAvz4RsyaGdI5lX66k7Pg0klO2rqT-xNzCwVuRv6KnESH0TIKDKE4vv9hjaEUwTPks4NjG59N-muuHebSbfPK7nDbABYfaB7yw")
model_local = MolecularModel().load_from_jaqpot(jaqpot=jaqpot, id="BKsEYKTVRSKZyCjEWBzp")

def bbb_function(mols):
    from jaqpotpy import Jaqpot
    from jaqpotpy.models import MolecularModel
    import numpy as np
    from jaqpotpy.cfg import config
    config.verbose = False
    rew = []

    for mol in mols:
        try:
            smiles = Chem.MolToSmiles(mol)
            model_local(smiles)
            # rew.append(model_local.prediction[0])
            rew.append(model_local.probability[0][1])
        except Exception as e:
            rew.append(0)
            continue
    return np.asarray(rew, dtype="float32")
    # model(["COc1cc(C=O)ccc1O"])
    # print(model.prediction)
    # print(model.probability)


def bbb_mean_function(mols):
    from jaqpotpy import Jaqpot
    from jaqpotpy.models import MolecularModel
    import numpy as np
    jaqpot = Jaqpot()
    jaqpot.set_api_key(
        "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJ3Ujh3X1lGOWpKWFRWQ2x2VHF1RkswZkctQXROQUJsb3FBd0N4MmlTTWQ4In0.eyJleHAiOjE2NzI0MDA1NTcsImlhdCI6MTY3MjIyNzc1NywiYXV0aF90aW1lIjoxNjcyMjI3NzU2LCJqdGkiOiJmZTA5NzliMi0yNzA1LTQxY2MtOTI5YS1jZjcwNmI0MzE0MWQiLCJpc3MiOiJodHRwczovL2xvZ2luLmphcXBvdC5vcmcvYXV0aC9yZWFsbXMvamFxcG90IiwiYXVkIjoiYWNjb3VudCIsInN1YiI6IjI0MjVkNzYwLTAxOGQtNDA4YS1hZTBiLWNkZTRjNTYzNTRiOSIsInR5cCI6IkJlYXJlciIsImF6cCI6ImphcXBvdC11aS1jb2RlIiwibm9uY2UiOiIzOGMwMTI2Njk1MWIwY2E5MjQ2YjEwOTUxMDk3YmY3NjcxVEpkYnpmaSIsInNlc3Npb25fc3RhdGUiOiJlMGVmMThkMC0yYjI1LTRjZDctYmZmZC02YTgxYTViNmY3YzYiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbIicqJyIsIioiXSwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6Im9wZW5pZCBqYXFwb3QtYWNjb3VudHMgZW1haWwgcHJvZmlsZSB3cml0ZSByZWFkIiwic2lkIjoiZTBlZjE4ZDAtMmIyNS00Y2Q3LWJmZmQtNmE4MWE1YjZmN2M2IiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJuYW1lIjoiUGFudGVsaXMgS2FyYXR6YXMiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJwYW50ZWxpc3BhbmthIiwiZ2l2ZW5fbmFtZSI6IlBhbnRlbGlzIiwiZmFtaWx5X25hbWUiOiJLYXJhdHphcyIsImVtYWlsIjoicGFudGVsaXNwYW5rYUBnbWFpbC5jb20ifQ.AZeIENjLUXKmChf7EUhQLgZ8qiDRR9Iwv-FsGrAmkGvdv9tCkGYBlNAFdL67YQr9XfFL2mx-SNRXNVK2Vmrbh1UrPj_tIJI9-dNfYqEpT4whlaAPUQqMpyO81RH-7oLqZupDeuMNTUHYM_FYThsCAcin7Fbgioax89U20UMwNBgsK0Eii6fWMFexyDB-MEXPiPa2IbBr8o246pjqFfohkTJNxoQwD3P5SNMaMMCxn1rzScsPlBdMZkUtB3BxmIqKXDKnpqvfgqJO2QryFo3X65G3KFeAdxtAp7mCkhqjrBJXX9IzRWpcTQZKJMl8mQvaSgJh758b0jurwp-_PVj2mA")
    model = MolecularModel().load_from_jaqpot(jaqpot=jaqpot, id="BKsEYKTVRSKZyCjEWBzp")
    rew = []
    for mol in mols:
        mol = Chem.MolToSmiles(mol)
        print(mol)
        if mol is None:
            rew.append([0])
        else:
            if type(mol).__name__ == 'str':
                mol = Chem.MolToSmiles(mol)
            try:
                model(mol)
                print(model.probability)
                print(model.probability[0][1])
                rew.append(model.probability[0][1])
            except Exception as e:
                print(str(e))
                rew.append(0)
    return np.asarray(sum(rew) / len(mols))
