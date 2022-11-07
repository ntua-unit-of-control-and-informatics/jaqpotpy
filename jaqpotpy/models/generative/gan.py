from jaqpotpy.models.generative.models import GanMoleculeGenerator, MoleculeDiscriminator
from jaqpotpy.datasets.molecular_datasets import SmilesDataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from jaqpotpy.descriptors.molecular import MolGanFeaturizer, GraphMatrix
from rdkit import Chem
from jaqpotpy.helpers.logging import init_logger
from typing import Any
import numpy as np
from jaqpotpy.models.generative.molecular_metrics import MolecularMetrics
from collections import defaultdict
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.info')


class GanSolver(object):

    def __init__(self, generator: GanMoleculeGenerator
                 , discriminator: MoleculeDiscriminator
                 , dataset: SmilesDataset

                 , g_lr
                 , d_lr
                 , lamda_gradient_penalty: float = 0.01
                 , la: float = 1.0
                 , n_critic: int = 1
                 , weight_decay: float = 5e-4
                 , optimizer: str = "RMSprop"
                 , batch_size: int = 32
                 , epochs: int = 12
                 , val_at: int = 10
                 , post_method: str = "softmax"
                 , metric='validity,qed'
                 , device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                 , path="./"
                 , task: str = "Generate / Reinforce"
                 # , OptimizationModel: MolecularModel
                 ):
        self.generator = generator
        self.discriminator = discriminator
        self.value_network = discriminator
        # self.generator = GanMoleculeGenerator([128, 256, 524], 8, 42, 5, 12, 0.5)
        # self.discriminator = MoleculeDiscriminator([[128, 64], 128, [128, 64]], 12, 5 - 1, 0.5)
        # self.value_network = MoleculeDiscriminator([[128, 64], 128, [128, 64]], 1, 5 - 1, 0.5)
        self.post_method = post_method
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

        self.metric = metric

        self.la = la
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.la_gp = lamda_gradient_penalty
        self.weight_decay = weight_decay

        self.optimizer = optimizer
        self.g_optim = None
        self.d_optim = None
        self.v_optim = None

        self.logger = init_logger(__name__, testing_mode=False, output_log_file=False)

        self.loader_params = {'batch_size': self.batch_size, 'shuffle': True, 'num_workers': 0}
        self.data_loader = DataLoader(self.dataset, **self.loader_params)

        self.val_after = val_at
        self.epochs = epochs
        self.num_steps = (len(self.dataset) // self.batch_size)

        if self.la > 0:
            self.n_critic = n_critic
        else:
            self.n_critic = 1

        self.build_env()

    def build_env(self):

        if self.optimizer == "Adam":
            self.g_optim = torch.optim.Adam(self.generator.parameters(), lr=self.g_lr, weight_decay=self.weight_decay)
            self.d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=self.g_lr,
                                            weight_decay=self.weight_decay)
            self.v_optim = torch.optim.Adam(self.value_network.parameters(), lr=self.g_lr,
                                            weight_decay=self.weight_decay)
        if self.optimizer == "AdamW":
            self.g_optim = torch.optim.AdamW(self.generator.parameters(), lr=self.g_lr, weight_decay=self.weight_decay)
            self.d_optim = torch.optim.AdamW(self.discriminator.parameters(), lr=self.g_lr,
                                             weight_decay=self.weight_decay)
            self.v_optim = torch.optim.AdamW(self.value_network.parameters(), lr=self.g_lr,
                                             weight_decay=self.weight_decay)
        if self.optimizer == "SGD":
            self.g_optim = torch.optim.SGD(self.generator.parameters(), lr=self.g_lr, weight_decay=self.weight_decay)
            self.d_optim = torch.optim.SGD(self.discriminator.parameters(), lr=self.g_lr,
                                           weight_decay=self.weight_decay)
            self.v_optim = torch.optim.SGD(self.value_network.parameters(), lr=self.g_lr,
                                           weight_decay=self.weight_decay)
        if self.optimizer == "Adadelta":
            self.g_optim = torch.optim.Adadelta(self.generator.parameters(), lr=self.g_lr,
                                                weight_decay=self.weight_decay)
            self.d_optim = torch.optim.Adadelta(self.discriminator.parameters(), lr=self.g_lr,
                                                weight_decay=self.weight_decay)
            self.v_optim = torch.optim.Adadelta(self.value_network.parameters(), lr=self.g_lr,
                                                weight_decay=self.weight_decay)
        if self.optimizer == "Adagrad":
            self.g_optim = torch.optim.Adagrad(self.generator.parameters(), lr=self.g_lr,
                                               weight_decay=self.weight_decay)
            self.d_optim = torch.optim.Adagrad(self.discriminator.parameters(), lr=self.g_lr,
                                               weight_decay=self.weight_decay)
            self.v_optim = torch.optim.Adagrad(self.value_network.parameters(), lr=self.g_lr,
                                               weight_decay=self.weight_decay)
        if self.optimizer == "Adamax":
            self.g_optim = torch.optim.Adamax(self.generator.parameters(), lr=self.g_lr, weight_decay=self.weight_decay)
            self.d_optim = torch.optim.Adamax(self.discriminator.parameters(), lr=self.g_lr,
                                              weight_decay=self.weight_decay)
            self.v_optim = torch.optim.Adamax(self.value_network.parameters(), lr=self.g_lr,
                                              weight_decay=self.weight_decay)
        if self.optimizer == "RMSprop":
            self.g_optim = torch.optim.RMSprop(self.generator.parameters(), lr=self.g_lr,
                                               weight_decay=self.weight_decay)
            self.d_optim = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.g_lr,
                                               weight_decay=self.weight_decay)
            self.v_optim = torch.optim.RMSprop(self.value_network.parameters(), lr=self.g_lr,
                                               weight_decay=self.weight_decay)
        if self.optimizer == "RMSprop":
            self.g_optim = torch.optim.RMSprop(self.generator.parameters(), lr=self.g_lr,
                                               weight_decay=self.weight_decay)
            self.d_optim = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.g_lr,
                                               weight_decay=self.weight_decay)
            self.v_optim = torch.optim.RMSprop(self.value_network.parameters(), lr=self.g_lr,
                                               weight_decay=self.weight_decay)

        self.discriminator.to(self.device)
        self.value_network.to(self.device)
        self.generator.to(self.device)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optim.zero_grad()
        self.d_optim.zero_grad()
        self.v_optim.zero_grad()

    def fit(self):

        the_step = self.num_steps
        for epoch_i in range(self.epochs):
            for a_step, i in enumerate(self.data_loader):

                nod_fs = torch.split(i[2][0], 1)
                adc_mat = torch.split(i[2][1], 1)
                mols = []
                for ind, n in enumerate(nod_fs):
                    g2 = GraphMatrix(torch.squeeze(adc_mat[ind]).cpu().detach().numpy(),
                                     torch.squeeze(n).cpu().detach().numpy())
                    m = self.dataset.featurizer.defeaturize(g2)
                    mols.append(m[0])

                if epoch_i < 0:
                    cur_la = 0
                else:
                    cur_la = self.la

                cur_step = self.num_steps * epoch_i + a_step

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                a_tensor = torch.permute(i[2][1], (0, 2, 3, 1))
                x_tensor = i[2][0]

                logits_real, features_real = self.discriminator(a_tensor, None, x_tensor)

                z = self.sample_z(i[2][0].size(dim=0))

                # samples = self.generator.sample_generator(self.data_loader.batch_size)
                edges_logits, nodes_logits = self.generator(z)
                (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)

                # in_adj.to(self.device)
                # in_feats.to(self.device)

                out_a_hat, logits_hat = self.discriminator(edges_hat, None, nodes_hat)

                # Compute losses for gradient penalty.
                eps = torch.rand(logits_real.size(0), 1, 1, 1).to(self.device)
                x_int0 = (eps * a_tensor + (1. - eps) * edges_hat).requires_grad_(True)
                x_int1 = (eps.squeeze(-1) * x_tensor + (1. - eps.squeeze(-1)) * nodes_hat).requires_grad_(True)
                grad0, grad1 = self.discriminator(x_int0, None, x_int1)
                grad_penalty = self.gradient_penalty(grad0, x_int0) + self.gradient_penalty(grad1, x_int1)

                d_loss_real = torch.mean(logits_real)
                d_loss_fake = torch.mean(logits_hat)
                loss_D = -d_loss_real + d_loss_fake + self.la_gp * grad_penalty

                if cur_step % self.n_critic != 0 and cur_la > 0:
                    self.reset_grad()
                    # self.d_optim.zero_grad()
                    loss_D.backward(inputs=list(self.discriminator.parameters()))
                    self.d_optim.step()

                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #

                edges_logits, nodes_logits = self.generator(z)
                (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
                logits_fake, features_fake = self.discriminator(edges_hat, None, nodes_hat)

                # logits_fake, features_fake = out_a_hat, logits_hat
                value_logit_real, _ = self.value_network(a_tensor, None, x_tensor, torch.sigmoid)
                value_logit_fake, _ = self.value_network(edges_hat, None, nodes_hat, torch.sigmoid)

                mols_hat = []

                nod_fs = torch.split(nodes_hat, 1)
                adc_mat = torch.split(edges_hat, 1)
                for ind, n in enumerate(nod_fs):
                    am = torch.squeeze(adc_mat[ind]).cpu()
                    am = torch.permute(am, (2, 0, 1))
                    g2 = GraphMatrix(am.detach().numpy(),
                                     torch.squeeze(n).cpu().detach().numpy())
                    t = self.dataset.featurizer.defeaturize(g2)
                    if t.size == 0:
                        mols_hat.append(None)
                    else:
                        mols_hat.append(t[0])

                # Real Reward
                reward_r = torch.from_numpy(self.reward(mols)).to(self.device)
                # Fake Reward
                reward_f = torch.from_numpy(self.reward(mols_hat)).to(self.device)

                # Losses Update
                loss_G = -logits_fake
                # Original TF loss_V. Here we use absolute values instead of the squared one.
                # loss_V = (value_logit_real - reward_r) ** 2 + (value_logit_fake - reward_f) ** 2
                loss_V = torch.abs(value_logit_real - reward_r) + torch.abs(value_logit_fake - reward_f)
                loss_RL = -value_logit_fake

                loss_G = torch.mean(loss_G)
                loss_V = torch.mean(loss_V)
                loss_RL = torch.mean(loss_RL)

                alpha = torch.abs(loss_G.detach() / loss_RL.detach()).detach()
                # alpha = torch.abs(loss_G / loss_RL)
                train_step_G = cur_la * loss_G + (1 - cur_la) * alpha * loss_RL
                #
                train_step_V = loss_V

                # alpha = torch.abs(loss_G.detach() / loss_RL.detach()).detach()
                # train_step_G = cur_la * loss_G.cpu().detach().numpy() + (1 - cur_la) * alpha.cpu().detach().numpy() * loss_RL.cpu().detach().numpy()
                # train_step_G = torch.from_numpy(np.full(1, train_step_G))
                # train_step_G.requires_grad = True
                #
                # train_step_V = loss_V.cpu().detach().numpy()
                # train_step_V = torch.from_numpy(np.full(1, train_step_V))
                # train_step_V.requires_grad = True

                self.logger.info("Generator loss for epoch "
                                 + str(epoch_i) + " step " + str(cur_step - self.num_steps * epoch_i)
                                 + ": " + str(loss_G.cpu().detach().numpy()))
                self.logger.info("Discriminator loss for epoch "
                                 + str(epoch_i) + " step " + str(cur_step - self.num_steps * epoch_i)
                                 + ": " + str(loss_V.cpu().detach().numpy()))
                self.logger.info("Value loss for epoch "
                                 + str(epoch_i) + " step " + str(cur_step - self.num_steps * epoch_i)
                                 + ": " + str(loss_RL.cpu().detach().numpy()))

                self.reset_grad()

                if cur_step % self.n_critic == 0:
                    self.g_optim.zero_grad()
                    train_step_G.backward(retain_graph=True, inputs=list(self.generator.parameters()))
                    self.g_optim.step()

                # Optimise value network.
                if cur_step % self.n_critic == 0:
                    self.v_optim.zero_grad()
                    train_step_V.backward(inputs=list(self.value_network.parameters()))
                    self.v_optim.step()

                losses = defaultdict(list)
                scores = defaultdict(list)

                if epoch_i > self.val_after and cur_step - self.num_steps * epoch_i == 0:
                    self.logger.info("VALIDATING")
                    mols = mols_hat
                    m0, m1 = self.all_scores(mols, mols, norm=True)  # 'mols' is output of Fake Reward
                    for k, v in m1.items():
                        scores[k].append(v)
                    for k, v in m0.items():
                        scores[k].append(np.array(v)[np.nonzero(v)].mean())

                    log = "Elapsed [{}], Iteration [{}/{}]:".format("undefined", epoch_i + 1, self.epochs)
                    is_first = True
                    for tag, value in losses.items():
                        if is_first:
                            log += "\n{}: {:.2f}".format(tag, np.mean(value))
                            is_first = False
                        else:
                            log += ", {}: {:.2f}".format(tag, np.mean(value))
                    is_first = True
                    for tag, value in scores.items():
                        if is_first:
                            log += "\n{}: {:.2f}".format(tag, np.mean(value))
                            is_first = False
                        else:
                            log += ", {}: {:.2f}".format(tag, np.mean(value))
                    print(log)

            pass

    def sample_z(self, batch_size):
        return torch.Tensor(np.random.normal(0, 1, size=(batch_size, 8)))

    def get_mols(self, data: Any):
        return ""

    def reward(self, mols):
        rr = 1.
        for m in ('logp,sas,qed,unique' if self.metric == 'all' else self.metric).split(','):

            if m == 'logp':
                rr *= MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
            # elif m == 'np':
            #     rr *= MolecularMetrics.natural_product_scores(mols, norm=True)
            # elif m == 'sas':
            # rr *= MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
            elif m == 'qed':
                rr *= MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True)
            elif m == 'novelty':
                rr *= MolecularMetrics.novel_scores(mols, self.data)
            elif m == 'dc':
                rr *= MolecularMetrics.drugcandidate_scores(mols, self.data)
            elif m == 'unique':
                rr *= MolecularMetrics.unique_scores(mols)
            elif m == 'diversity':
                rr *= MolecularMetrics.diversity_scores(mols, self.data)
            elif m == 'validity':
                rr *= MolecularMetrics.valid_scores(mols)
            else:
                raise RuntimeError('{} is not defined as a metric'.format(m))

        return rr.reshape(-1, 1)

    def postprocess(self, inputs, method, temperature=1.):

        def listify(x):
            return x if type(x) == list or type(x) == tuple else [x]

        def delistify(x):
            return x if len(x) > 1 else x[0]

        if method == 'soft_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
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

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def print_network(self, model, name, log=None):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        if log is not None:
            self.logger.log(model)
            self.logger.log(name)
            self.logger.log("The number of parameters: {}".format(num_params))
        else:
            print(model)
            print(name)
            print("The number of parameters: {}".format(num_params))

    def all_scores(self, mols, data, norm=False, reconstruction=False):
        m0 = {k: list(filter(lambda e: e is not None, v)) for k, v in {
            'NP': MolecularMetrics.natural_product_scores(mols, norm=norm),
            'QED': MolecularMetrics.quantitative_estimation_druglikeness_scores(mols),
            'Solute': MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=norm),
            # 'SA': MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=norm),
            'diverse': MolecularMetrics.diversity_scores(mols, data),
            'drugcand': MolecularMetrics.drugcandidate_scores(mols, data)}.items()}

        m1 = {'valid': MolecularMetrics.valid_total_score(mols) * 100,
              'unique': MolecularMetrics.unique_total_score(mols) * 100,
              'novel': MolecularMetrics.novel_total_score(mols, data) * 100}

        return m0, m1
