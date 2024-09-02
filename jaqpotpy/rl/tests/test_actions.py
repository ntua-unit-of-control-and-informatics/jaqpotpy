import unittest
from jaqpotpy.rl.spaces import Action, Actions
from rdkit import Chem


class TestActions(unittest.TestCase):
    @unittest.skip("RL has not been tested yet in the newest version of jaqpotpy")
    def test_actions_on_mol(self):
        m = Chem.MolFromSmiles("CC(=O)C=CC=C")
        mw = Chem.RWMol(m)
        mw.AddAtom(Chem.Atom(6))

    @unittest.skip("RL has not been tested yet in the newest version of jaqpotpy")
    def test_actions(self):
        acts = []
        action1 = Action(0, "Add atom C")
        action2 = Action(1, "Add atom O")
        action2.action = 1
        action2.action_meta = "Add atom O"
        action3 = Action(2, "Add bond")
        action4 = Action(3, "remove bond")
        acts.append(action1)  # = [action1, action2, action3, action4]
        acts.append(action2)
        acts.append(action3)
        acts.append(action4)

        actions = Actions
        actions.actions = acts

        for act in acts:
            print(act.action)
            print(act.action_meta)

        for act in actions.actions:
            print(act.action)
