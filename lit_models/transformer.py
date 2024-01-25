import json
import sys

import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

from .base import BaseLitModel
from .util import f1_eval, compute_f1, acc, f1_score
from transformers.optimization import get_linear_schedule_with_warmup
from functools import partial
import random
from sklearn.mixture import GaussianMixture
from torch.distributions.multivariate_normal import MultivariateNormal
import time
import math
import torch.distributions as dist
import torch.nn.functional as F
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class RobertaLitModel(BaseLitModel):
    """
    use AutoModelForMaskedLM, and select the output by another layer in the lit model
    """
    def __init__(self, model, args, tokenizer):
        super().__init__(model, args)
        self.tokenizer = tokenizer
        
        with open(f"{args.data_dir}/rel2id.json", "r") as file:
            rel2id = json.load(file)
        
        Na_num = 0
        for k, v in rel2id.items():
            if k == "NA" or k == "no_relation" or k == "Other":
                Na_num = v
                break
        self.num_relation = len(rel2id)
        # init loss function
        self.loss_fn = nn.CrossEntropyLoss()
        # ignore the no_relation class to compute the f1 score
        self.eval_fn = partial(f1_score, rel_num=self.num_relation, na_num=Na_num)
        self.best_f1 = 0
        self.label_st_id = tokenizer("[class1]", add_special_tokens=False)['input_ids'][0]
        self.tokenizer = tokenizer
        self._init_label_word()
        self.inputdata_reduced = []
        self.alldata_reduced = []
        self.current_datasetsplit = args.data_dir.split("/")[-1]
        self.current_datasetname = args.data_dir.split("/")[1]

    def _init_label_word(self, ):
        args = self.args
        # ./dataset/dataset_name
        dataset_name = args.data_dir.split("/")[1]
        model_name_or_path = args.model_name_or_path.split("/")[-1]
        label_path = f"./dataset/{model_name_or_path}_{dataset_name}.pt"
        # [num_labels, num_tokens], ignore the unanswerable

        label_word_idx = torch.load(label_path)    # label_word_idx:19*4
        num_labels = len(label_word_idx)
        self.model.resize_token_embeddings(len(self.tokenizer))
        with torch.no_grad():
            word_embeddings = self.model.get_input_embeddings()
            continous_label_word = [a[0] for a in self.tokenizer([f"[class{i}]" for i in range(1, num_labels+1)], add_special_tokens=False)['input_ids']]   #class1-class19 ——> tokenizer id  list type
            if self.args.init_answer_words:
                if self.args.init_answer_words_by_one_token:
                    for i, idx in enumerate(label_word_idx):
                        word_embeddings.weight[continous_label_word[i]] = word_embeddings.weight[idx][-1]
                else:
                    for i, idx in enumerate(label_word_idx):
                        word_embeddings.weight[continous_label_word[i]] = torch.mean(word_embeddings.weight[idx], dim=0)

            assert torch.equal(self.model.get_input_embeddings().weight, word_embeddings.weight)
            assert torch.equal(self.model.get_input_embeddings().weight, self.model.get_output_embeddings().weight)
        self.word2label = continous_label_word # a continous list
            
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels, so = batch

        # # ============== save all data e.g. semeval.sh data_dir=dataset/semeval batchsize=1 ==============
        # if self.current_epoch == 0:
        #     print("epoch==0:store all data")
        #     self.store_alldata(input_ids)
        # else:
        #     print("The process has exited.")
        #     sys.exit()
        # return None

        # # ============== save updated data or normal ==============
        # if self.current_epoch == 0:
        #     # print("epoch==0: store input data")
        #     self.store_inputdata(input_ids)
        # else:
        #     sys.exit()

        # =============== after saving updated data: train ===============
        # print("self.current_datasetname:", self.current_datasetname)
        # print("self.current_datasetsplit:", self.current_datasetsplit)
        # load updated data
        data_document = 'updated_{}'.format(self.current_datasetname)
        data_file = 'updated_{}_{}.txt'.format(self.current_datasetname, self.current_datasetsplit)
        updated_data_filename = os.path.join(data_document, data_file)
        # print("updated_data_filename:", updated_data_filename)
        self.theta = np.loadtxt(updated_data_filename)
        if self.current_epoch > 0:
            self.add_sample_abstraction(input_ids)
        result = self.model(input_ids, attention_mask, return_dict=True, output_hidden_states=True)
        logits = result.logits
        logits = self.pvp(logits, input_ids)
        loss = self.loss_fn(logits, labels)
        self.log("Train/loss", loss)
        return loss

    # def training_epoch_end(self, outputs):
    #     # # ============= save all data =============
    #     # if self.current_epoch == 0:
    #     #     # print("epoch==0 end: save all data as .txt")
    #     #     save_alldata_reduced = torch.stack(self.alldata_reduced).detach().cpu()  # batchsize*1*1024
    #     #     save_alldata_reduced = save_alldata_reduced.reshape(save_alldata_reduced.shape[0], -1)  # batchsize*1024
    #     #     print("save_alldata_reduced:", save_alldata_reduced.shape)
    #     #     # store save_alldata_reduced.txt
    #     #     save_alldata_reduced_name = 'all_{}.txt'.format(self.current_datasetname)
    #     #     np.savetxt(save_alldata_reduced_name, save_alldata_reduced.numpy(), delimiter=' ')
    #     #     print("Success！")
    #     #     sys.exit()
    #
    #     # # ============= save updated data or normal=============
    #     if self.current_epoch == 0:
    #         start_time = time.time()
    #         print("epoch == 0 end: gmm fit all data")
    #         all_train_data_name = "all_{}.txt".format(self.current_datasetname)
    #         all_train_data_reduced = np.loadtxt(all_train_data_name)  # data_num*1024
    #         print("all_train_data_reduced:", all_train_data_name, all_train_data_reduced.shape)  # 6507*1024
    #         gmm_means, gmm_covariances, gmm_weights, gmm = self.get_gmm(all_train_data_reduced)
    #         gradient_GMM = gmm_gradient(gmm_means, gmm_covariances, gmm_weights, gmm)
    #         print("epoch == 0 end: process store_inputdata")
    #         all_inputdata_reduced = torch.stack(self.inputdata_reduced).detach().cpu()
    #         all_inputdata_reduced = all_inputdata_reduced.reshape(all_inputdata_reduced.shape[0], -1)  # current_date_num * 1024
    #         print("all_inputdata_reduced:", all_inputdata_reduced.shape)
    #         # updated particles ——> self.theta
    #         self.theta = SVGD().update(all_inputdata_reduced.cuda(), gradient_GMM.dlnprob, n_iter=1000, stepsize=0.01)
    #         print("self.theta:", self.theta.shape) # current_date_num * 1024
    #
    #         # store updated theta
    #         updated_theta_filename = 'updated_{}_{}.txt'.format(self.current_datasetname, self.current_datasetsplit)
    #         np.savetxt(updated_theta_filename, self.theta.detach().cpu().numpy(), delimiter=' ')
    #         print("Success！")
    #         end_time = time.time()
    #         elapsed_time = end_time - start_time
    #         print("The execution time of gmm_svgd is: ", elapsed_time, "s")
    #         sys.exit()

    def get_gmm(self, all_inputdata_reduced):
        gmm = GaussianMixture(n_components=min(all_inputdata_reduced.shape[0], self.num_relation), reg_covar=1e-1)
        gmm.fit(all_inputdata_reduced)
        means = torch.tensor(gmm.means_).cuda()
        # print("means:", means, means.shape) # n_components*1024
        covariances = torch.tensor(gmm.covariances_).cuda()
        # print("covariances:", covariances, covariances.shape) # n_components*1024*1024
        weights = torch.tensor(gmm.weights_).cuda()
        # print("weights:", weights, weights.shape) # n_components
        # probs = torch.tensor(gmm.predict_proba(all_inputdata_reduced)).cuda()
        # print("probs:", probs, probs.shape)
        # logP_x = torch.tensor(gmm.score_samples(all_inputdata_reduced)).cuda()
        # print("logP_x:", logP_x, logP_x.shape)
        # P_x = torch.exp(logP_x)
        # print("P_x:", P_x, P_x.shape)
        return means, covariances, weights, gmm

    def add_sample_abstraction(self, input_ids):
        word_embeddings = self.model.get_input_embeddings()
        # [sub]:50288 [obj]:50289
        so_word = [a[0] for a in self.tokenizer(["[sub]", "[obj]"], add_special_tokens=False)['input_ids']]
        # get updated particles
        particles_data = self.theta
        particles_data_list = particles_data.tolist()
        for i in range(input_ids.shape[0]):
            sample_num = min(len(particles_data_list), self.num_relation)
            sampled_data = random.sample(particles_data_list, sample_num)
            sampled_data = torch.tensor(sampled_data)
            avg_pool = nn.AvgPool1d(kernel_size=sample_num)  # input_ids.shape[1]=input_embedding.shape[0]=256
            sampled_data_avg = avg_pool(sampled_data.T).T
            with torch.no_grad():
                word_embeddings.weight[so_word[0]] = sampled_data_avg
                word_embeddings.weight[so_word[1]] = sampled_data_avg

    def store_inputdata(self, input_ids):
        word_embeddings = self.model.get_input_embeddings()  # roberta-large Embedding(50295, 1024)
        avg_pool = nn.AvgPool1d(kernel_size=input_ids.shape[1])  # input_ids.shape[1]=input_embedding.shape[0]=256
        for i in range(input_ids.shape[0]):
            input_embedding = word_embeddings.weight[input_ids[i]]  # 256*1024
            self.inputdata_reduced.append(avg_pool(input_embedding.T).T)  # 1*1024

    def store_alldata(self, input_ids):
        word_embeddings = self.model.get_input_embeddings()  # roberta-large Embedding(50295, 1024)
        avg_pool = nn.AvgPool1d(kernel_size=input_ids.shape[1])  # input_ids.shape[1]=input_embedding.shape[0]=256
        for i in range(input_ids.shape[0]):
            input_embedding = word_embeddings.weight[input_ids[i]]  # 256*1024
            self.alldata_reduced.append(avg_pool(input_embedding.T).T)  # 1*1024

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels, entities = batch  # input_ids:[16,256]  attention_mask:[16,256]   labels:[16]  entities:[16,4]
        if self.current_epoch > 0:
            self.add_sample_abstraction(input_ids)
        logits = self.model(input_ids, attention_mask, return_dict=True).logits
        logits = self.pvp(logits, input_ids)
        loss = self.loss_fn(logits, labels)
        self.log("Eval/loss", loss)
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy()}

    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])
        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Eval/f1", f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels, _ = batch
        if self.current_epoch > 0:
            self.add_sample_abstraction(input_ids)
        logits = self.model(input_ids, attention_mask, return_dict=True).logits
        logits = self.pvp(logits, input_ids)
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Test/f1", f1)

    def pvp(self, logits, input_ids):
        # convert the [batch_size, seq_len, vocab_size] => [batch_size, num_labels]
        # ! hard coded
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)  # mask_token_id:50264
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        assert mask_idx.shape[0] == bs, "only one mask in sequence!"
        final_output = mask_output[:, self.word2label]
        return final_output

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]
        if not self.args.two_steps:
            parameters = self.model.named_parameters()
        else:
            parameters = [next(self.model.named_parameters())]
        parameters = list(parameters)
        # only optimize the embedding parameters
        optimizer_group_parameters = [
            {"params": [p for n, p in parameters if not any(nd in n for nd in no_decay_param)],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in parameters if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]
        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * 0.1,
                                                    num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }

class gmm_gradient:
    def __init__(self, means, covariances, weights, gmm):
        self.means = means
        self.covariances = covariances
        self.weights = weights
        self.gmm = gmm

    def dlnprob(self, theta):
        # probs = torch.tensor(self.gmm.predict_proba(theta.detach().cpu())).cuda()
        # print("probs:", probs, probs.shape) # current_data_num * n_components
        logP_x = torch.tensor(self.gmm.score_samples(theta.detach().cpu())).cuda()
        # print("logP_x:", logP_x, logP_x.shape) # current_data_num
        P_x = torch.exp(logP_x)
        # print("P_x:", P_x, P_x.shape) # current_data_num
        K = len(self.weights)
        dxlogPx = torch.zeros_like(theta)
        # avoid /0
        epsilon = 1e-10
        P_x_p = P_x + epsilon
        for i in range(K):
            # dxlogMVN_x: g(x)~MVN，▽xlogP(x)=?
            dxlogMVN_x = -torch.matmul((theta - self.means[i].view(1, self.means[i].shape[0])),
                                       torch.inverse(self.covariances[i]))
            m = MultivariateNormal(self.means[i], self.covariances[i])
            pi_multi_gx_dev_px = self.weights[i] * torch.exp(m.log_prob(theta)) / P_x_p
            reshape_pi_multi_gx_dev_px = pi_multi_gx_dev_px.reshape((pi_multi_gx_dev_px.shape[0], -1))
            dxlogPx += reshape_pi_multi_gx_dev_px * dxlogMVN_x
        return dxlogPx

class SVGD():
    def __init__(self):
        pass

    def pairwise_distances(self, x):
        dot_product = torch.matmul(x, x.t())
        square_norm = dot_product.diag()
        distances = square_norm.unsqueeze(1) - 2.0 * dot_product + square_norm.unsqueeze(0)
        distances[distances < 0] = 0
        return distances

    def svgd_kernel(self, theta, h=-1):
        pairwise_dists = self.pairwise_distances(theta)
        if h < 0:  # if h < 0, using median trick
            h = torch.median(pairwise_dists)
            h = torch.sqrt(0.5 * h / torch.log(torch.tensor([theta.shape[0] + 1.0]).cuda()))

        # compute the rbf kernel
        Kxy = torch.exp(-pairwise_dists / h ** 2 / 2)
        dxkxy = -torch.matmul(Kxy, theta)
        sumkxy = torch.sum(Kxy, dim=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:, i] + theta[:, i] * sumkxy
        dxkxy = dxkxy / (h ** 2)
        return (Kxy, dxkxy)

    def update(self, x0, lnprob, n_iter=1000, stepsize=1e-3, bandwidth=-1, alpha=0.9, debug=False):
        # Check input
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')

        theta = x0.clone()

        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        for iter in range(n_iter):
            if debug and (iter + 1) % 1000 == 0:
                print('iter ' + str(iter + 1))

            lnpgrad = lnprob(theta)
            # calculating the kernel matrix
            kxy, dxkxy = self.svgd_kernel(theta, h=-1)
            grad_theta = (torch.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0]

            # adagrad
            if iter == 0:
                historical_grad = historical_grad + grad_theta ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
            adj_grad = grad_theta / (fudge_factor + torch.sqrt(historical_grad))
            theta = theta + stepsize * adj_grad
        return theta

class TransformerLitModelTwoSteps(RobertaLitModel):
    def configure_optimizers(self):
        no_decay_param = ["bais", "LayerNorm.weight"]
        parameters = list(self.model.named_parameters())
        optimizer_group_parameters = [
            {"params": [p for n, p in parameters if not any(nd in n for nd in no_decay_param)],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in parameters if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]
        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.args.lr_2, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * 0.1,
                                                    num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }