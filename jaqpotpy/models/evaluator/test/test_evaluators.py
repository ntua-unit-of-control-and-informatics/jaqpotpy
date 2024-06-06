import unittest
from jaqpotpy.jaqpot import Jaqpot

from jaqpotpy.models import MolecularModel
from jaqpotpy.models.evaluator import GenerativeEvaluator
from jaqpotpy.models.generative.molecular_metrics import diversity_scores \
    , drugcandidate_scores, synthetic_accessibility_score_scores, valid_mean \
    , quantitative_estimation_druglikeness_scores, novel_score \
    , water_octanol_partition_coefficient_scores, unique_total_score, valids, valid_scores
# pylint: disable=no-member


class TestEvaluators(unittest.TestCase):

    def setUp(self) -> None:
        self.jaqpot = Jaqpot()
        self.jaqpot.set_api_key(
            "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJ3Ujh3X1lGOWpKWFRWQ2x2VHF1RkswZkctQXROQUJsb3FBd0N4MmlTTWQ4In0.eyJleHAiOjE2NjY1MjU0MzQsImlhdCI6MTY2NjM1MjYzNCwiYXV0aF90aW1lIjoxNjY2MzUyNjM0LCJqdGkiOiIxODk2ZWQ3ZC1lMmM1LTQ2ZWItOTE3MS1mYzQyYWRiYzVkMmUiLCJpc3MiOiJodHRwczovL2xvZ2luLmphcXBvdC5vcmcvYXV0aC9yZWFsbXMvamFxcG90IiwiYXVkIjoiYWNjb3VudCIsInN1YiI6IjI0MjVkNzYwLTAxOGQtNDA4YS1hZTBiLWNkZTRjNTYzNTRiOSIsInR5cCI6IkJlYXJlciIsImF6cCI6ImphcXBvdC11aS1jb2RlIiwibm9uY2UiOiI0NDAxMTk1NDUyNGQzMTk5ZGIxNDZkYjAwMWViZGZiYzNlbFFSS2JwWiIsInNlc3Npb25fc3RhdGUiOiIzMGVlZTc1YS03NDg2LTRjN2MtYjI1Ni1kNmM0ZjIxOGI2MDQiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbIicqJyIsIioiXSwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6Im9wZW5pZCBqYXFwb3QtYWNjb3VudHMgZW1haWwgcHJvZmlsZSB3cml0ZSByZWFkIiwic2lkIjoiMzBlZWU3NWEtNzQ4Ni00YzdjLWIyNTYtZDZjNGYyMThiNjA0IiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJuYW1lIjoiUGFudGVsaXMgS2FyYXR6YXMiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJwYW50ZWxpc3BhbmthIiwiZ2l2ZW5fbmFtZSI6IlBhbnRlbGlzIiwiZmFtaWx5X25hbWUiOiJLYXJhdHphcyIsImVtYWlsIjoicGFudGVsaXNwYW5rYUBnbWFpbC5jb20ifQ.GFFG1HIKcYsdVqAFSavL_1SgLN0GJWffL3yOahUV_4h2JmqYx5LkzELAk9EfLDroqzxclcl0uBEutrSKIPK2-PSN-8cl_4UY-9__E2F6iG7OR_wOtW5vOY54L5p8LqC0nnbrck_U59sVz9k-zzIsSgmV1NJwDxduL8nzN5FQBjnUW8ZAsMXx1AhK23-x3KPe2L1s2lAq4JE7azXL7ahu7tdSAaKsJRzF0ZdSwUCWic_awc5nNxjspoToK-OrC8-0cDx1cDgX30-L1Wy2gTMBzjW4gHkQ-uhKZ9-yVa_bWK162x0JRGzE_RzZf9DiPebE-jCShxTcuK_8pw1YfoILyw")
        self.mols_mocking_genereted = [
            "CCOc1cccc(NC(=O)NCc2ccc(N3CCSCC3)cc2)c1",
            "Cc1cccc(NC(=O)CN2CCN(c3ccc4c(c3)OCCO4)C2=O)n1",
            "C=C(C)C(=O)N[C@H](C)c1nc2ccccc2n1CCC(=O)N1CCCCCC1",
            "CCOC[C@H]1CC[NH+](Cc2ccc(-c3nc4ccccc4s3)o2)C1",
            "CCOC(=O)[C@]1(Cc2cccc(Cl)c2)CCCN(C(=O)c2ccnn2C)C1",
            "Cc1ccc([N+](=O)[O-])cc1NC(=O)C(=O)N1CC[C@H]([NH+]2CCCC2)C1",
            "CCOCCCNC(=O)N[C@@H]1CCC[C@@H](CC)C1",
            "O=C(Cc1cccc(F)c1F)Nc1cccc(Br)n1",
            "COc1ccccc1NC(=O)[C@@H]1CCCN(C(=O)Nc2cccs2)C1",
            "C[C@H]1CCC[C@](C#N)([C@]2(O)CCCCC2(C)C)C1",
            "O=C(NCc1ccc([N+]2=CCCC2)cc1)NC1(c2ccc(Cl)cc2)CC1",
            "CCCC(=O)N[C@@H]1CCC[NH+](Cc2ncccc2C)C1",
            "O=C(NCc1cccs1)C1(c2cccc(Cl)c2)CCC1",
            "C[C@H]1CCC[C@H](NC(=O)[C@@H](C)Sc2ncn[nH]2)[C@@H]1C",
            "COc1ccc([C@@H]([NH2+]Cc2ccc(Cl)nc2)c2ccc(F)cc2)cc1",
            "COc1cc(NC(=O)[C@H](C)Sc2ccccc2Cl)cc(OC)c1",
            "CCN1CCC(=NNC(=O)c2ccccc2)CC1",
            "CCCOc1ccc(Br)cc1C[NH+]1CCC([C@@H](C)O)CC1",
            "Cc1cc2n(C[C@H](O)CO[C@H](c3ccccc3)c3ccccc3C)c(=O)c3ccccc3n2n1",
            "CCc1ncc(CN(C)C(=O)Nc2c(C)ccc([N+](=O)[O-])c2C)s1",
            "c1ccc2nc(NCCCc3nc4ccccc4[nH]3)cnc2c1",
            "Cc1c([C@H](C)[NH2+]Cc2cccn2C)cnn1C",
            "CC(=O)N[C@@H](C(=O)NC1COC1)C(C)C",
            "O=C(Nc1ccccc1F)c1cc2ccccc2c2cccnc12",
            "Cc1ccccc1N1C(=O)/C(=C/c2cccn2-c2cccc([N+](=O)[O-])c2)C([O-])=NC1=S",
            "COCCN1C[C@H](C(=O)N(Cc2cccc(Cl)c2)C(C)C)CC1=O",
            "COc1ccc(NC(=O)N2CCN(C(=O)Cc3csc4ccccc34)CC2)cc1OC",
            "C#CCN(C[C@H]1CCCO1)C(=O)N[C@@H](C)c1cccc([N+](=O)[O-])c1",
            "Cc1cccc(C2=CCN(C(=O)Nc3ccc(C(N)=O)c(Cl)c3)CC2)c1",
            "COCCCN1C(=O)c2ccc(C(=O)Nc3nc(-c4ccc(C)cc4)cs3)cc2C1=O",
            "O=S(=O)(Nc1ccc(N2CCCS2(=O)=O)cc1)c1ccc(F)c(Cl)c1",
            "O=C1/C(=C/c2ccccc2)Oc2c1ccc1c2CN(Cc2cccs2)CO1",
            "CC[C@@H](C)[C@@H](NC(=O)c1cccc(F)c1)C(=O)N=c1[nH]c2ccccc2[nH]1",
            "Cc1c(F)cc(N)cc1S(=O)(=O)N[C@@H](C)C1CC1",
            "Cc1ccc(Cn2ncc3c(N)ncnc32)cc1",
            "CCOC(=O)C(C)(C)c1nc(-c2ccccc2)no1",
            "CCOC(=O)c1sc(/C=C/c2nc3c(s2)CCC3)nc1C",
            "C[C@@H]1CC[C@@H]([NH2+]C2CCC(NS(C)(=O)=O)CC2)c2ccccc21",
            "CN(C)S(=O)(=O)c1ccc(C(=O)N(C(=O)N2CCCCC2)c2ccccc2)cc1",
            "CC[NH2+][C@@H](CC)c1ccccc1OCc1cccc(F)c1",
            "C=CCOC(=O)C1=C(C)N=C2S[C@H](C)C(=O)N2[C@H]1c1ccc(F)cc1",
            "CCC[NH2+][C@]1(C(=O)OCC)CC[C@H](n2cc(Cl)c(C)n2)C1",
            "Cc1cc(NC(=O)CSc2nnc3c4ccccc4n(C)c3n2)ccc1Br",
            "CCOC(=O)c1sc(NC(=O)c2ccc(-n3c(C)nc4ccccc4c3=O)cc2)cc1C",
            "CC#CCC(=O)C1([NH+](CC)CC)CCCC1",
            "O=C(NCCCc1ccccc1)C1CCN(C(=O)[C@@H]2CC(=O)N(c3ccccc3)C2)CC1",
            "O=Cc1ccc(OCc2ccn(-c3cccc(F)c3)n2)cc1",
            "Clc1ccccc1Cn1ccnc1",
                     ]


        self.mols_no_valids = [
            "CCOc1ccccAdsfvSdSCC3)cc2)c1",
            "Cc1cccc(NC(=O)CN2CCN(c3ccc4c(c3)OCCO4)C2=O)n1",
            "C=C(C)C(=O)N[C@H](C)c1nc2ccccc2n1CCC(=O)N1CCCCCC1",
            "CCOC[C@H]1CC[NH+](Cc2ccc(-c3nc4ccccc4s3)o2)C1",
            "CCOC(=O)[C@]1(Cc2cccc(Cl)c2)CCCN(C(=O)c2ccnn2C)C1",
            "Cc1ccc([N+](=O)[O-])cc1NC(=O)C(=O)N1CC[C@H]([NH+]2CCCC2)C1",
            "CCOCCCNC(=O)N[C@@H]1CCC[C@@H](CC)C1",
            "O=C(Cc1cccc(F)c1F)Nc1cccc(Br)n1",
            "COc1ccccc1NC(=O)[C@@H]1CCCN(C(=O)Nc2cccs2)C1",
            "C[C@H]1CCC[C@](C#N)([C@]2(O)CCCCC2(C)C)C1",
            "O=C(NCc1ccc([N+]2=CCCC2)cc1)NC1(c2ccc(Cl)cc2)CC1",
            "CCCC(=O)N[C@@H]1CCC[NH+](Cc2ncccc2C)C1",
            "O=C(NCc1cccs1)C1(c2cccc(Cl)c2)CCC1",
            "C[C@H]1CCC[C@H](NC(=O)[C@@H](C)Sc2ncn[nH]2)[C@@H]1C",
            "COc1ccc([C@@H]([NH2+]Cc2ccc(Cl)nc2)c2ccc(F)cc2)cc1",
            "COc1cc(NC(=O)[C@H](C)Sc2ccccc2Cl)cc(OC)c1",
            "CCN1CCCadsfijsldc2)CC1",
            "CCCOc1ccc(Br)cc1C[NH+]1CCC([C@@H](C)O)CC1",
            "Cc1cc2n(C[C@H](O)CO[C@H](c3ccccc3)c3ccccc3C)c(=O)c3ccccc3n2n1",
            "CCc1ncc(CN(C)C(=O)Nc2c(C)ccc([N+](=O)[O-])c2C)s1",
            "c1ccc2nc(NCCCc3nc4ccccc4[nH]3)cnc2c1",
            "Cc1c([C@H](C)[NH2+]Cc2cccn2C)cnn1C",
            "CC(=O)N[C@@H](C(C)C",
            "",
            "Cc1ccccc1N1C(=O)/C(=C/c2cccn2-c2cccc([N+](=O)[O-])c2)C([O-])=NC1=S",
            "COCCN1C[C@H](C(=O)N(Cc2cccc(Cl)c2)C(C)C)CC1=O",
            "COc1ccc(NC(=O)N2CCN(C(=O)Cc3csc4ccccc34)CC2)cc1OC",
            "C#CCN(C[C@H]1CCCO1)C(=O)N[C@@H](C)c1cccc([N+](=O)[O-])c1",
            "Cc1cccc(C2=CCN(C(=O)Nc3ccc(C(N)=O)c(Cl)c3)CC2)c1",
            "COCCCN1C(=O)c2ccc(C(=O)Nc3nc(-c4ccc(C)cc4)cs3)cc2C1=O",
            "O=S(=O)(Nc1ccc(N2CCCS2(=O)=O)cc1)c1ccc(F)c(Cl)c1",
            "O=C1/C(=C/c2ccccc2)Oc2c1ccc1c2CN(Cc2cccs2)CO1",
            "CC[C@@H](C)[C@@H](NC(=O)c1cccc(F)c1)C(=O)N=c1[nH]c2ccccc2[nH]1",
            "Cc1c(F)cc(N)cc1S(=O)(=O)N[C@@H](C)C1CC1",
            "Cc1ccc(Cn2ncc3c(N)ncnc32)cc1",
            "CCOC(=O)C(C)(C)c1nc(-c2ccccc2)no1",
            "CCOC(=O)c1sc(/C=C/c2nc3c(s2)CCC3)nc1C",
            "C[C@@H]1CC[C@@H]([NH2+]C2CCC(NS(C)(=O)=O)CC2)c2ccccc21",
            "CN(C)S(=O)(=O)c1ccc(C(=O)N(C(=O)N2CCCCC2)c2ccccc2)cc1",
            "CC[NH2+][C@@H](CC)c1ccccc1OCc1cccc(F)c1",
            "C=CCOC(=O)C1=C(C)N=C2S[C@H](C)C(=O)N2[C@H]1c1ccc(F)cc1",
            "CCC[NH2+][C@]1(C(=O)OCC)CC[C@H](n2cc(Cl)c(C)n2)C1",
            "Cc1cc(NC(=O)CSc2nnc3c4ccccc4n(C)c3n2)ccc1Br",
            "CCOC(=O)c1sc(NC(=O)c2ccc(-n3c(C)nc4ccccc4c3=O)cc2)cc1C",
            "CC#CCC(=O)C1([NH+](CC)CC)CCCC1",
            "O=C(NCCCc1ccccc1)C1CCN(C(=O)[C@@H]2CC(=O)N(c3ccccc3)C2)CC1",
            "O=Cc1ccc(OCc2ccn(-c3cccc(F)c3)n2)cc1",
            "Clc1ccccc1Cn1ccnc1",
                     ]


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

    # def test_load_jaqpot_model(self):
    #     model = MolecularModel.load_from_jaqpot(jaqpot=self.jaqpot, id="iaFePw8jwhHSbzg42acr")
    #     print(model.library)
    #     print(model.version)
    #     import torch
    #     print(torch.__version__)
    #     import torch_geometric
    #     print(torch_geometric.__version__)
    #     model(self.mols)
    #     print(model.prediction)

    @unittest.skip("This test needs refactoring")
    def test_generative_evaluator(self):
        from rdkit import Chem
        gen_eval = GenerativeEvaluator()
        gen_eval.register_scoring_function("Valid all", valid_mean)
        gen_eval.register_scoring_function("QED", quantitative_estimation_druglikeness_scores)
        gen_eval.register_scoring_function("Synthetic Accessibility", synthetic_accessibility_score_scores)
        gen_eval.register_scoring_function("Novel", novel_score)
        gen_eval.register_scoring_function("Unique", unique_total_score)
        # gen_eval.register_scoring_function("Diversity", diversity_scores)
        gen_eval.register_scoring_function("Water Oct", water_octanol_partition_coefficient_scores)
        gen_eval.register_scoring_function("Drugcandidate Scores", drugcandidate_scores)

        # gen_eval.register_scoring_function("Novel all", novel_score)
        gen_eval.register_dataset(self.mols_test)
        mols = []
        for i in self.mols_mocking_genereted:
            mols.append(Chem.MolFromSmiles(i))
        rew = gen_eval.get_reward(mols)
        print(rew)

    @unittest.skip("This test needs refactoring")
    def test_generative_evaluator_no_valids(self):
        from rdkit import Chem
        gen_eval = GenerativeEvaluator()
        gen_eval.register_scoring_function("Valid all", valid_mean)
        gen_eval.register_scoring_function("QED", quantitative_estimation_druglikeness_scores)
        gen_eval.register_scoring_function("Synthetic Accessibility", synthetic_accessibility_score_scores)
        gen_eval.register_scoring_function("Novel", novel_score)
        gen_eval.register_scoring_function("Unique", unique_total_score)
        # gen_eval.register_scoring_function("Diversity", diversity_scores)
        gen_eval.register_scoring_function("Water Oct", water_octanol_partition_coefficient_scores)
        gen_eval.register_scoring_function("Drugcandidate Scores", drugcandidate_scores)

        # gen_eval.register_scoring_function("Novel all", novel_score)
        gen_eval.register_dataset(self.mols_test)
        mols = []
        for i in self.mols_no_valids:
            try:
                mol = Chem.MolFromSmiles(i)
                mols.append(mol)
            except Exception as e:
                mols.append(None)
        rew = gen_eval.get_reward(mols)

    @unittest.skip("This test needs refactoring")
    def test_generative_evaluator_scores(self):
        from rdkit import Chem
        gen_eval = GenerativeEvaluator()
        gen_eval.register_scoring_function("Valid all", valid_mean)
        gen_eval.register_scoring_function("QED", quantitative_estimation_druglikeness_scores)
        gen_eval.register_scoring_function("Synthetic Accessibility", synthetic_accessibility_score_scores)
        gen_eval.register_scoring_function("Novel", novel_score)
        gen_eval.register_scoring_function("Unique", unique_total_score)
        gen_eval.register_scoring_function("Diversity", diversity_scores)
        gen_eval.register_scoring_function("Water Oct", water_octanol_partition_coefficient_scores)
        gen_eval.register_scoring_function("Drugcandidate Scores", drugcandidate_scores)

        # gen_eval.register_scoring_function("Novel all", novel_score)
        gen_eval.register_dataset(self.mols_test)
        mols = []
        for i in self.mols_no_valids:
            try:
                mol = Chem.MolFromSmiles(i)
                mols.append(mol)
            except Exception as e:
                mols.append(None)
        rew = gen_eval.get_reward(mols)

    @unittest.skip("This test needs refactoring")
    def test_generative_evaluator_scores_valids(self):
        from rdkit import Chem
        gen_eval = GenerativeEvaluator()
        gen_eval.register_scoring_function("Valids", valid_scores)
        gen_eval.register_scoring_function("QED", quantitative_estimation_druglikeness_scores)

        # gen_eval.register_scoring_function("Novel all", novel_score)
        gen_eval.register_dataset(self.mols_test)
        mols = []
        for i in self.mols_no_valids:
            try:
                mol = Chem.MolFromSmiles(i)
                mols.append(mol)
            except Exception as e:
                mols.append(None)
        rew = gen_eval.get_reward(mols)




# gen_eval = GenerativeEvaluator()
#
#
# def jaqpotpy_activity(mols):
#     from jaqpotpy.models import MolecularModel
#     from jaqpotpy.jaqpot import Jaqpot
#     jaqpot = Jaqpot()
#     jaqpot.login("pantelispanka", "kapan2")
#     model = MolecularModel().load_from_jaqpot("u60obQ1Y23e7GL2SQsYe")
#     model(mols)
#     pred = model.prediction
#     percentage = 0.99
#     return percentage
#
# gen_eval.register_scoring_function("Jaqpotpy_covid_activity", jaqpotpy_activity)
#
#
# def jaqpotpy_tox(mols):
#     from jaqpotpy.models import MolecularModel
#     tox_model = MolecularModel().load_from_jaqpot("Jaqpotid")
#     tox_model(mols)
#     pred = tox_model.prediction
#     percentage = 0.60
#     return percentage
#
# gen_eval.register_scoring_function("jaqpotpy_tox", jaqpotpy_tox)
# gen_eval.get_reward(mols="")
#
#
# def jaqpotpy_pipeline(mols):
#     from jaqpotpy.models import MolecularModel
#     model = MolecularModel().load_from_jaqpot("Jaqpotid")
#     tox_model = MolecularModel().load_from_jaqpot("Jaqpotid")
#
# from jaqpotpy.models.generative.molecular_metrics import novel_score
# gen_eval.register_scoring_function("novel", novel_score)
