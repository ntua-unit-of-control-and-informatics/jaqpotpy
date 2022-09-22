from jaqpotpy.models.generative.models import GanMoleculeGenerator, GanMoleculeDiscriminator
from jaqpotpy.datasets.molecular_datasets import SmilesDataset
import torch

from jaqpotpy.helpers.logging import init_logger


class GanSolver(object):

    def __init__(self, generator: GanMoleculeGenerator
                 , discriminator: GanMoleculeDiscriminator
                 , dataset: SmilesDataset
                 , batch_size: int = 32
                 , post_method: str = "hard_gumbel"
                 , device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                 , task: str = "Generate / Reinforce"
                 # , OptimizationModel: MolecularModel
                 ):
        self.generator = generator
        self.discriminator = discriminator
        self.value_network = discriminator
        self.post_method = post_method
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.logger = self.log = init_logger(__name__, testing_mode=False, output_log_file=False)

    def fit(self):
        self.logger.log("A MESSAGE")
        pass

    def postprocess(self, inputs, method, temperature=1.):

        def listify(x):
            return x if type(x) == list or type(x) == tuple else [x]

        def delistify(x):
            return x if len(x) > 1 else x[0]

        if method == 'soft_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1,e_logits.size(-1))
                       / temperature, hard=False).view(e_logits.size())
                       for e_logits in listify(inputs)]
        elif method == 'hard_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                       / temperature, hard=True).view(e_logits.size())
                       for e_logits in listify(inputs)]
        else:
            softmax = [F.softmax(e_logits / temperature, -1)
                       for e_logits in listify(inputs)]

        return [delistify(e) for e in (softmax)]