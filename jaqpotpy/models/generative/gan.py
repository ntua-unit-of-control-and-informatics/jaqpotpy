import datetime
import os
import time

import rdkit

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
from jaqpotpy.models.evaluator import GenerativeEvaluator
from collections import defaultdict
from rdkit import RDLogger

from jaqpotpy.models.generative.molecular_metrics import MolecularMetrics

RDLogger.DisableLog('rdApp.info')


class GanSolver(object):

    def __init__(self, generator: GanMoleculeGenerator
                 , discriminator: MoleculeDiscriminator
                 , dataset: SmilesDataset
                 , evaluator: GenerativeEvaluator
                 , g_lr
                 , d_lr
                 , lamda_gradient_penalty: float = 20.0
                 , la: float = 0.5
                 , n_critic: int = 3
                 , weight_decay: float = 5e-4
                 , optimizer: str = "Adam"
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
        self.start_time = time.time()
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

        self.evaluator = evaluator

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

        self.image_dir_path = path

        if self.la > 0:
            self.n_critic = n_critic
        else:
            self.n_critic = 1

        self.BOND_DIM = self.dataset.featurizer.BOND_DIM
        self.MAX_ATOMS = self.dataset.featurizer.MAX_ATOMS
        self.build_env()

    def build_env(self):

        if self.optimizer == "Adam":
            self.g_optim = torch.optim.Adam(self.generator.parameters(), lr=self.g_lr, weight_decay=self.weight_decay)
            self.d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=self.d_lr,
                                            weight_decay=self.weight_decay)
            self.v_optim = torch.optim.Adam(self.value_network.parameters(), lr=self.d_lr,
                                            weight_decay=self.weight_decay)
        if self.optimizer == "AdamW":
            self.g_optim = torch.optim.AdamW(self.generator.parameters(), lr=self.g_lr, weight_decay=self.weight_decay)
            self.d_optim = torch.optim.AdamW(self.discriminator.parameters(), lr=self.d_lr,
                                             weight_decay=self.weight_decay)
            self.v_optim = torch.optim.AdamW(self.value_network.parameters(), lr=self.d_lr,
                                             weight_decay=self.weight_decay)
        if self.optimizer == "SGD":
            self.g_optim = torch.optim.SGD(self.generator.parameters(), lr=self.g_lr, weight_decay=self.weight_decay)
            self.d_optim = torch.optim.SGD(self.discriminator.parameters(), lr=self.d_lr,
                                           weight_decay=self.weight_decay)
            self.v_optim = torch.optim.SGD(self.value_network.parameters(), lr=self.d_lr,
                                           weight_decay=self.weight_decay)
        if self.optimizer == "Adadelta":
            self.g_optim = torch.optim.Adadelta(self.generator.parameters(), lr=self.g_lr,
                                                weight_decay=self.weight_decay)
            self.d_optim = torch.optim.Adadelta(self.discriminator.parameters(), lr=self.d_lr,
                                                weight_decay=self.weight_decay)
            self.v_optim = torch.optim.Adadelta(self.value_network.parameters(), lr=self.d_lr,
                                                weight_decay=self.weight_decay)
        if self.optimizer == "Adagrad":
            self.g_optim = torch.optim.Adagrad(self.generator.parameters(), lr=self.g_lr,
                                               weight_decay=self.weight_decay)
            self.d_optim = torch.optim.Adagrad(self.discriminator.parameters(), lr=self.d_lr,
                                               weight_decay=self.weight_decay)
            self.v_optim = torch.optim.Adagrad(self.value_network.parameters(), lr=self.d_lr,
                                               weight_decay=self.weight_decay)
        if self.optimizer == "Adamax":
            self.g_optim = torch.optim.Adamax(self.generator.parameters(), lr=self.g_lr, weight_decay=self.weight_decay)
            self.d_optim = torch.optim.Adamax(self.discriminator.parameters(), lr=self.d_lr,
                                              weight_decay=self.weight_decay)
            self.v_optim = torch.optim.Adamax(self.value_network.parameters(), lr=self.d_lr,
                                              weight_decay=self.weight_decay)
        if self.optimizer == "RMSprop":
            self.g_optim = torch.optim.RMSprop(self.generator.parameters(), lr=self.g_lr,
                                               weight_decay=self.weight_decay)
            self.d_optim = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.d_lr,
                                               weight_decay=self.weight_decay)
            self.v_optim = torch.optim.RMSprop(self.value_network.parameters(), lr=self.d_lr,
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
                mol_strings = []
                for ind, n in enumerate(nod_fs):
                    g2 = GraphMatrix(torch.squeeze(adc_mat[ind]).cpu().detach().numpy(),
                                     torch.squeeze(n).cpu().detach().numpy())
                    m = self.dataset.featurizer._defeaturize(g2, sanitize=False, cleanup=False)
                    mol_strings.append(Chem.MolToSmiles(m))
                    mols.append(m)

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

                edges_logits, nodes_logits = self.generator(z)
                (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)

                logits_fake, features_fake = self.discriminator(edges_hat, None, nodes_hat)

                # Compute losses for gradient penalty.
                eps = torch.rand(logits_real.size(0), 1, 1, 1).to(self.device)
                x_int0 = (eps * a_tensor + (1. - eps) * edges_hat).requires_grad_(True)
                x_int1 = (eps.squeeze(-1) * x_tensor + (1. - eps.squeeze(-1)) * nodes_hat).requires_grad_(True)
                grad0, grad1 = self.discriminator(x_int0, None, x_int1)
                grad_penalty = self.gradient_penalty(grad0, x_int0) + self.gradient_penalty(grad1, x_int1)

                d_loss_real = torch.mean(logits_real)
                d_loss_fake = torch.mean(logits_fake)
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

                    adjacency = torch.argmax(adc_mat[ind], dim=3)
                    adjacency = torch.nn.functional.one_hot(adjacency, num_classes=self.BOND_DIM)
                    adjacency = torch.squeeze(adjacency)
                    adjacency = torch.permute(adjacency, (2, 0, 1))

                    nodes = torch.argmax(n, dim=2)
                    nodes = torch.nn.functional.one_hot(nodes, num_classes=self.MAX_ATOMS)

                    g2 = GraphMatrix(adjacency.detach().numpy(),
                                     torch.squeeze(nodes).cpu().detach().numpy())

                    t = self.dataset.featurizer.defeaturize(g2)
                    if t.size == 0:
                        mols_hat.append(None)
                    else:
                        if t[0] is not None:
                            print(Chem.MolToSmiles(t[0]))
                        mols_hat.append(t[0])

                # Real Reward
                reward_r = torch.from_numpy(self.evaluator.get_reward(mols)).to(self.device)
                # Fake Reward
                reward_f = torch.from_numpy(self.evaluator.get_reward(mols_hat)).to(self.device)

                # # Real Reward (OLD MOLECULAR METRICS)
                # reward_r = torch.from_numpy(self.reward(mols)).to(self.device)
                # # Fake Reward (OLD MOLECULAR METRICS)
                # reward_f = torch.from_numpy(self.reward(mols_hat)).to(self.device)

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

                self.logger.info("Reward from reals for epoch "
                      + str(epoch_i) + " step " + str(cur_step - self.num_steps * epoch_i)
                      + ": " + str(np.mean(reward_r.cpu().detach().numpy())))
                self.logger.info("Reward from fakes for epoch "
                      + str(epoch_i) + " step " + str(cur_step - self.num_steps * epoch_i)
                      + ": " + str(np.mean(reward_f.cpu().detach().numpy())))
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
                    # self.g_optim.zero_grad()
                    train_step_G.backward(retain_graph=True, inputs=list(self.generator.parameters()))
                    self.g_optim.step()

                # Optimise value network.
                if cur_step % self.n_critic == 0:
                    # self.v_optim.zero_grad()
                    train_step_V.backward(inputs=list(self.value_network.parameters()))
                    self.v_optim.step()

                losses = defaultdict(list)
                scores = defaultdict(list)

                if epoch_i > self.val_after and cur_step - self.num_steps * epoch_i == 0:
                    self.logger.info("VALIDATING")
                    mols = mols_hat
                    self.all_scores(mols, self.evaluator.dataset, epoch_i=epoch_i, norm=True)
                    # self.all_scores(mols, self.evaluator.dataset, epoch_i=epoch_i, norm=True)  # 'mols' is output of Fake Reward
            pass

    def sample_z(self, batch_size):
        return torch.Tensor(np.random.normal(0, 1, size=(batch_size, 8)))

    def get_mols(self, data: Any):
        return ""

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

    def all_scores_(self, mols, data, epoch_i, norm=False, reconstruction=False):
        m0 = {k: list(filter(lambda e: e is not None, v)) for k, v in {
                'NP': MolecularMetrics.natural_product_scores(mols, norm=norm),
                'QED': MolecularMetrics.quantitative_estimation_druglikeness_scores(mols),
                'Solute': MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=norm),
                # 'SA': MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=norm),
                # 'diverse': MolecularMetrics.diversity_scores(mols, data),
                'drugcand': MolecularMetrics.drugcandidate_scores(mols, data)}.items()}

        m1 = {'valid': MolecularMetrics.valid_total_score(mols) * 100,
                  'unique': MolecularMetrics.unique_total_score(mols) * 100,
                  'novel': MolecularMetrics.novel_total_score(mols, data) * 100}
        losses = defaultdict(list)
        scores = defaultdict(list)
        for k, v in m1.items():
            scores[k].append(v)
        for k, v in m0.items():
            scores[k].append(np.array(v)[np.nonzero(v)].mean())

            # Saving molecule images.
        mol_f_name = os.path.join(self.image_dir_path, 'mol-{}.png'.format(epoch_i))
        save_mol_img(mols, mol_f_name)

        # Print out training information.
        et = time.time() - self.start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        log = "Elapsed [{}], Iteration [{}/{}]:".format(et, epoch_i + 1, self.epochs)

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
        if self.logger is not None:
            self.logger.info(log)

    def all_scores(self, mols, data, epoch_i, norm=False, reconstruction=False):
        et = time.time() - self.start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        log = "Elapsed [{}], Iteration [{}/{}]:".format(et, epoch_i + 1, self.epochs)
        for key in self.evaluator.eval_functions.keys():
            function_name = key
            f = self.evaluator.eval_functions.get(key)
            try:
                score = f(mols)
                log += ", {}: {:.2f}".format(function_name, np.mean(score))
            except TypeError as e:
                score = f(mols, data)
                log += ", {}: {:.2f}".format(function_name, np.mean(score))
        if self.logger is not None:
            self.logger.info(log)
        else:
            print(log)
        mol_f_name = os.path.join(self.image_dir_path, 'mol-{}.png'.format(epoch_i))
        save_mol_img(mols, mol_f_name)

    def reward(self, mols):
        rr = 1.
        for m in ('logp,sas,qed,unique' if self.metric == 'all' else self.metric).split(','):

            if m == 'np':
                rr *= MolecularMetrics.natural_product_scores(mols, norm=True)
            elif m == 'logp':
                rr *= MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
            elif m == 'sas':
                rr *= MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
            elif m == 'qed':
                rr *= MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True)
            elif m == 'novelty':
                rr *= MolecularMetrics.novel_scores(mols, self.evaluator.dataset)
            elif m == 'dc':
                rr *= MolecularMetrics.drugcandidate_scores(mols, self.evaluator.dataset)
            elif m == 'unique':
                rr *= MolecularMetrics.unique_scores(mols)
            elif m == 'diversity':
                rr *= MolecularMetrics.diversity_scores(mols, self.evaluator.dataset)
            elif m == 'validity':
                rr *= MolecularMetrics.valid_scores(mols)
            else:
                raise RuntimeError('{} is not defined as a metric'.format(m))

        return rr.reshape(-1, 1)


import datetime
import string
import random

def save_mol_img(mols, f_name='tmp.png', is_test=False):
    orig_f_name = f_name
    for a_mol in mols:
        try:
            if Chem.MolToSmiles(a_mol) is not None:
                print('Generating molecule')

                if is_test:
                    f_name = orig_f_name
                    f_split = f_name.split('.')
                    f_split[-1] = random_string() + '.' + f_split[-1]
                    f_name = ''.join(f_split)

                rdkit.Chem.Draw.MolToFile(a_mol, f_name)
                break

                # if not is_test:
                #     break
        except:
            continue



def random_string(string_len=3):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_len))