import unittest
from jaqpotpy.jaqpot import Jaqpot

from jaqpotpy.models import MolecularModel
from jaqpotpy.models.evaluator import GenerativeEvaluator


class TestEvaluators(unittest.TestCase):

    def setUp(self) -> None:
        self.jaqpot = Jaqpot()
        self.jaqpot.set_api_key("eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJ3Ujh3X1lGOWpKWFRWQ2x2VHF1RkswZkctQXROQUJsb3FBd0N4MmlTTWQ4In0.eyJleHAiOjE2NjY1MjU0MzQsImlhdCI6MTY2NjM1MjYzNCwiYXV0aF90aW1lIjoxNjY2MzUyNjM0LCJqdGkiOiIxODk2ZWQ3ZC1lMmM1LTQ2ZWItOTE3MS1mYzQyYWRiYzVkMmUiLCJpc3MiOiJodHRwczovL2xvZ2luLmphcXBvdC5vcmcvYXV0aC9yZWFsbXMvamFxcG90IiwiYXVkIjoiYWNjb3VudCIsInN1YiI6IjI0MjVkNzYwLTAxOGQtNDA4YS1hZTBiLWNkZTRjNTYzNTRiOSIsInR5cCI6IkJlYXJlciIsImF6cCI6ImphcXBvdC11aS1jb2RlIiwibm9uY2UiOiI0NDAxMTk1NDUyNGQzMTk5ZGIxNDZkYjAwMWViZGZiYzNlbFFSS2JwWiIsInNlc3Npb25fc3RhdGUiOiIzMGVlZTc1YS03NDg2LTRjN2MtYjI1Ni1kNmM0ZjIxOGI2MDQiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbIicqJyIsIioiXSwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6Im9wZW5pZCBqYXFwb3QtYWNjb3VudHMgZW1haWwgcHJvZmlsZSB3cml0ZSByZWFkIiwic2lkIjoiMzBlZWU3NWEtNzQ4Ni00YzdjLWIyNTYtZDZjNGYyMThiNjA0IiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJuYW1lIjoiUGFudGVsaXMgS2FyYXR6YXMiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJwYW50ZWxpc3BhbmthIiwiZ2l2ZW5fbmFtZSI6IlBhbnRlbGlzIiwiZmFtaWx5X25hbWUiOiJLYXJhdHphcyIsImVtYWlsIjoicGFudGVsaXNwYW5rYUBnbWFpbC5jb20ifQ.GFFG1HIKcYsdVqAFSavL_1SgLN0GJWffL3yOahUV_4h2JmqYx5LkzELAk9EfLDroqzxclcl0uBEutrSKIPK2-PSN-8cl_4UY-9__E2F6iG7OR_wOtW5vOY54L5p8LqC0nnbrck_U59sVz9k-zzIsSgmV1NJwDxduL8nzN5FQBjnUW8ZAsMXx1AhK23-x3KPe2L1s2lAq4JE7azXL7ahu7tdSAaKsJRzF0ZdSwUCWic_awc5nNxjspoToK-OrC8-0cDx1cDgX30-L1Wy2gTMBzjW4gHkQ-uhKZ9-yVa_bWK162x0JRGzE_RzZf9DiPebE-jCShxTcuK_8pw1YfoILyw")
        self.mols = ['O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1'
            , 'O=C1CCc2cc(C(=O)N3CCC(C4CCNC4)CC3)ccc2N1'
            , 'CCC(=O)Nc1ccc(N(Cc2ccccc2)C(=O)n2nnc3ccccc32)cc1'
            , 'COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1'
            , 'Cc1nn(C)c2[nH]nc(NC(=O)Cc3cccc(Cl)c3)c12'
            , 'O=C(Cc1cncc2ccccc12)N(CCC1CCCCC1)c1cccc(Cl)c1'
            , 'COc1ccc(N(Cc2ccccc2)C(=O)Cc2c[nH]c3ccccc23)cc1'
            , 'CC(C)(C)c1ccc(N(C(=O)c2ccco2)[C@H](C(=O)NCCc2cccc(F)c2)c2cccnc2)cc1'
            , 'OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2'
            , 'Cc1ccncc1NC(=O)Cc1cc(Cl)cc(-c2cnn(C)c2C(F)F)c1'
            , 'Cc1cc(C(F)(F)F)nc2c1c(N)nn2C(=O)Cc1cccc(Cl)c1'
            , 'Cc1cc(C(F)(F)F)nc2c1c(N)nn2C(=O)C1CCOc2ccc(Cl)cc21'
            , 'O=C(c1cc(=O)[nH]c2ccccc12)N1CCN(c2cccc(Cl)c2)C(=O)C1'
            , 'O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12'
            , 'COCCNC(=O)[C@@H](c1ccccc1)N1Cc2ccccc2C1=O'
            , 'CNCC1CCCN(C(=O)[C@@H](c2ccccc2)N2Cc3ccccc3C2=O)C1'
            , 'O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12'
            , 'COc1ccc2c(NC(=O)C3CCOc4ccc(Cl)cc43)[nH]nc2c1'
            , 'O=C(NC1N=Nc2ccccc21)C1CCOc2ccc(Cl)cc21'
            , 'COc1ccccc1OC1CCN(C(=O)c2cc(=O)[nH]c3ccccc23)C1'
            , 'O=C(Cc1cc(Cl)cc(Cc2ccn[nH]2)c1)Nc1cncc2ccccc12'
            , 'CN(C)c1ccc(N(Cc2ccsc2)C(=O)Cc2cncc3ccccc23)cc1'
            , 'C[C@H]1COc2ccc(Cl)cc2[C@@H]1C(=O)Nc1cncc2ccccc12'
                ]

    def test_load_jaqpot_model(self):
        model = MolecularModel.load_from_jaqpot(jaqpot=self.jaqpot, id="iaFePw8jwhHSbzg42acr")
        print(model.library)
        print(model.version)
        import torch
        print(torch.__version__)
        import torch_geometric
        print(torch_geometric.__version__)
        model(self.mols)
        print(model.prediction)




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
