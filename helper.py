from shutil import copyfile
import datetime
import math
import torch

from torch.autograd import Variable
import logging
import numpy as np
import copy
import random
from torch.nn.functional import log_softmax
import torch.nn.functional as F
import os
from copy import deepcopy
import sklearn.datasets as data
import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from models.Function import CosSim, CosSim_with_key

torch.manual_seed(1)
torch.cuda.manual_seed(1)

random.seed(0)
np.random.seed(0)

class Helper:
    def __init__(self, params):
        self.target_model = None
        self.local_model = None

        self.train_data = None
        self.benign_test_data = None
        self.poisoned_data = None
        self.poisoned_test_data = None

        self.params = params
        self.best_loss = math.inf

    @staticmethod
    def get_weight_difference(weight1, weight2):
        difference = {}
        res = []
        if type(weight2) == dict:
            for name, layer in weight1.items():
                difference[name] = layer.data - weight2[name].data
                res.append(difference[name].view(-1))
        else:
            for name, layer in weight2:
                difference[name] = weight1[name].data - layer.data
                res.append(difference[name].view(-1))

        difference_flat = torch.cat(res)

        return difference, difference_flat

    @staticmethod
    def get_l2_norm(weight1, weight2):
        difference = {}
        res = []
        if type(weight2) == dict:
            for name, layer in weight1.items():
                difference[name] = layer.data - weight2[name].data
                res.append(difference[name].view(-1))
        else:
            for name, layer in weight2:
                difference[name] = weight1[name].data - layer.data
                res.append(difference[name].view(-1))

        difference_flat = torch.cat(res)

        l2_norm = torch.norm(difference_flat.clone().detach().cuda())
        #l2_norm = torch.norm(difference_flat.clone().detach().cuda(), float('inf'))

        l2_norm_np = np.linalg.norm(difference_flat.cpu().numpy())
        #l2_norm_np = np.linalg.norm(difference_flat.cpu().numpy(), ord=np.inf)

        return l2_norm, l2_norm_np

    @staticmethod
    def clip_grad(norm_bound, weight_difference, difference_flat):
        l2_norm = torch.norm(difference_flat.clone().detach().cuda())
        print('l2_norm : ', l2_norm)
        #l2_norm = torch.norm(difference_flat.clone().detach().cuda(), float('inf'))
        scale =  max(1.0, float(torch.abs(l2_norm / norm_bound)))
        print('norm scale : ', scale)
        for name in weight_difference.keys():
            weight_difference[name].div_(scale)
            # weight_difference[name] /= scale

        return weight_difference, l2_norm

    def grad_mask(self, helper, model, dataset_clearn, criterion, ratio=0.5):
        """Generate a gradient mask based on the given dataset"""
        model.train()
        model.zero_grad()
        hidden = model.init_hidden(helper.params['batch_size'])
        for participant_id in range(len(dataset_clearn)):
            train_data = dataset_clearn[participant_id]
            if helper.params['task'] == 'word_predict':
                data_iterator = range(0, train_data.size(0) - 1, helper.params['sequence_length'])
                ntokens = 50000
                for batch in data_iterator:
                    model.train()
                    data, targets = helper.get_batch(train_data, batch)
                    hidden = helper.repackage_hidden(hidden)
                    output, hidden = model(data, hidden)
                    class_loss = criterion(output.view(-1, ntokens), targets)
                    class_loss.backward(retain_graph=True)
            elif helper.params['task'] == 'sentiment':
                for inputs, labels in train_data:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    hidden = helper.repackage_hidden(hidden)
                    inputs = inputs.type(torch.LongTensor).cuda()
                    output, hidden = model(inputs, hidden)
                    loss = criterion(output.squeeze(), labels.float())
                    loss.backward(retain_graph=True)
            else:
                raise ValueError("Unkonwn task")
        mask_grad_list = []

        # Next Token Prediction
        if helper.params['aggregate_all_layer'] == 1:
            grad_list = []
            print('pgd : ', helper.params['PGD'])
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    grad_list.append(parms.grad.abs().view(-1))
            grad_list = torch.cat(grad_list).cuda()
            count = 0
            for layer_name, parms in model.named_parameters():
                print('---------start--------------')
                print('layer name : ', layer_name)

                if helper.params['masking'] == True:
                    print('ih ratio : ', helper.params['ih'])
                    print('hh ratio : ', helper.params['hh'])
                if parms.requires_grad:
                    mask_flat = np.zeros( count + len(parms.grad.abs().view(-1))  )
                    print('mask_flat_shape : ', mask_flat.shape)
                    grad_flat = parms.grad.abs().view(-1)
                    print('grad_flat shape : ', grad_flat.shape)

                    if layer_name == 'rnn.weight_ih_l0' or layer_name == 'rnn.weight_ih_l1':
                        _, topk_indices = torch.topk(-1*grad_flat, int(len(grad_flat) * helper.params['ih']))
                        print('[correct] ', layer_name)
                        mask_flat[count + topk_indices.cpu().numpy()] = 1
                        print('inner mask_flat shape : ', mask_flat.shape)

                        ones_count = np.count_nonzero(mask_flat[count:count+len(parms.grad.abs().view(-1))])
                        zeros_count = len(mask_flat[count:count+len(parms.grad.abs().view(-1))]) - ones_count
                        print(f'count 1 : {ones_count}, count 0 : {zeros_count}')

                    elif layer_name == 'rnn.weight_hh_l0' or layer_name == 'rnn.weight_hh_l1':
                        _, topk_indices = torch.topk(-1*grad_flat, int(len(grad_flat) * helper.params['hh']))
                        print('[correct] ', layer_name)
                        mask_flat[count + topk_indices.cpu().numpy()] = 1
                        print('inner mask_flat shape : ', mask_flat.shape)

                        ones_count = np.count_nonzero(mask_flat[count:count+len(parms.grad.abs().view(-1))])
                        zeros_count = len(mask_flat[count:count+len(parms.grad.abs().view(-1))]) - ones_count
                        print(f'count 1 : {ones_count}, count 0 : {zeros_count}')

                    mask_flat = mask_flat[count:count + len(parms.grad.abs().view(-1))]
                    mask = list(mask_flat.reshape(parms.grad.abs().size()))

                    mask = torch.from_numpy(np.array(mask, dtype='float32')).cuda()
                    mask_grad_list.append(mask)
                    count += len(parms.grad.abs().view(-1))

        # Sentiment Analysis
        else:
            for layer_name, parms in model.named_parameters():
                print('---------start--------------')
                print('layer name : ', layer_name)
                if parms.requires_grad:
                    gradients = parms.grad.abs().view(-1)
                    gradients_length = len(gradients)
                    mask_flat = torch.zeros(gradients_length)
                    #---------if
                    if layer_name == 'lstm.weight_ih_l0' or layer_name == 'lstm.weight_ih_l1':
                        _, topk_indices = torch.topk(-1*gradients, int(gradients_length * helper.params['ih']))
                        print('[correct] ', layer_name)
                        mask_flat[topk_indices.cpu().numpy()] = 1
                        print('inner mask_flat shape : ', mask_flat.shape)
                        ones_count = np.count_nonzero(mask_flat)
                        zeros_count = len(mask_flat) - ones_count
                        print(f'count 1 : {ones_count}, count 0 : {zeros_count}')

                    if layer_name == 'lstm.weight_hh_l0' or layer_name == 'lstm.weight_hh_l1':
                        _, topk_indices = torch.topk(-1*gradients, int(gradients_length * helper.params['hh']))
                        print('[correct] ', layer_name)
                        mask_flat[topk_indices.cpu().numpy()] = 1
                        print('inner mask_flat shape : ', mask_flat.shape)
                        ones_count = np.count_nonzero(mask_flat)
                        zeros_count = len(mask_flat) - ones_count
                        print(f'count 1 : {ones_count}, count 0 : {zeros_count}')
                    
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())
        model.zero_grad()
        return mask_grad_list


    def grad_mask_gpt2(self, helper, model, dataset_clearn, criterion, ratio=0.5):
        """Generate a gradient mask based on the given dataset"""
        model.train()
        model.zero_grad()
        for i in range(len(dataset_clearn)):
            print('defense : ',helper.params['defense'])
            print('pgd : ', helper.params['PGD'])
            train_dataloader = dataset_clearn[i]
            for batch_id, batch in enumerate(train_dataloader):
                model.train()

                data1, data2 = batch['input_ids'], batch['attention_mask']
                # data1, data2 = data1.cuda(), data2.cuda()

                data1 = [x.unsqueeze(0) for x in data1]
                data2 = [x.unsqueeze(0) for x in data2]

                data1 = torch.cat(data1).transpose(0,1)
                data2 = torch.cat(data2).transpose(0,1)

                input_ids = data1[:,0:0+helper.params['sequence_length']]
                att_masks = data2[:,0:0+helper.params['sequence_length']]

                target = data1[:,1:1+helper.params['sequence_length']].reshape(-1)

                input_ids, att_masks, target = input_ids.cuda(), att_masks.cuda(), target.cuda()

                output = model(input_ids, attention_mask=att_masks).logits

                loss = criterion(output.contiguous().view(-1, self.n_tokens), target)
                loss.backward(retain_graph=True)

                ######## debug:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        mask_grad_list = []

        for layer_name, parms in model.named_parameters():
            #print('--------------start-----------------')
            print('layer name : ', layer_name)
            if parms.requires_grad:
                gradients = parms.grad.abs().view(-1)
                gradients_length = len(gradients)
                mask_flat = torch.zeros(gradients_length)
                
                if layer_name == 'transformer.h.0.mlp.c_fc.weight' or layer_name == 'transformer.h.1.mlp.c_fc.weight' or\
                        layer_name == 'transformer.h.2.mlp.c_fc.weight' or layer_name == 'transformer.h.3.mlp.c_fc.weight' or \
                        layer_name == 'transformer.h.4.mlp.c_fc.weight' or layer_name == 'transformer.h.5.mlp.c_fc.weight' or \
                        layer_name == 'transformer.h.6.mlp.c_fc.weight' or layer_name == 'transformer.h.7.mlp.c_fc.weight' or \
                        layer_name == 'transformer.h.8.mlp.c_fc.weight' or layer_name == 'transformer.h.9.mlp.c_fc.weight' or \
                        layer_name == 'transformer.h.10.mlp.c_fc.weight' or layer_name == 'transformer.h.11.mlp.c_fc.weight':
                    _, topk_indices = torch.topk(-1*gradients, int(gradients_length * helper.params['mlp_fc']))
                    print('mlp_fc : ', helper.params['mlp_fc'])
                    print('[correct] ', layer_name)
                    mask_flat[topk_indices.cpu().numpy()] = 1
                    print('inner mask_flat shape : ', mask_flat.shape)
                    ones_count = np.count_nonzero(mask_flat)
                    zeros_count = len(mask_flat) - ones_count
                    print(f'count 1 : {ones_count}, count 0 : {zeros_count}')

                mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())

        model.zero_grad()
        return mask_grad_list

    def lr_decay(self, epoch):
        # return 1 * (0.995 ** epoch)
        # if self.params['dataset'] == 'IMDB':
        #     return 0.1
        if self.params['model'] == 'GPT2':
            return 10
            
        return 1
        # return 1
        # return 1 - (epoch - 1) / self.params['end_epoch']
        # return 1 / math.sqrt(epoch + 1)
        # return max(1 - (epoch - 1) / 250, 0.05)

    @staticmethod
    def dp_noise(param, sigma=0.001):
        print('sigma : ', sigma)

        noised_layer = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)

        return noised_layer

    def average_shrink_models(self, weight_accumulator, target_model, epoch, wandb):
        """
        Perform FedAvg algorithm and perform some clustering on top of it.

        """
        lr = self.lr_decay(epoch)
        wandb.log({ 'global lr': lr, 'epoch': epoch})
        for name, data in target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                print('skipping')
                continue
            update_per_layer = weight_accumulator[name] * \
                               (1/self.params['partipant_sample_size']) * \
                               lr
            update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)

            update_per_layer = update_per_layer.cuda()
            if self.params['diff_privacy']:
                print('diff_privacy on!')
                if 'LongTensor' in update_per_layer.type():
                    pass
                else:
                    if self.params['model'] == 'GPT2':
                        update_per_layer.add_(self.dp_noise(data, sigma = 0.00001).cuda())
                    else:
                        update_per_layer.add_(self.dp_noise(data).cuda())

            data.add_(update_per_layer)

        return True

    def average_shrink_models_multi_krum(self, weight_accumulator, target_model, l, epoch, wandb):
        """
        Perform FedAvg algorithm and perform some clustering on top of it.

        """
        print('l : ', l)
        lr = self.lr_decay(epoch)
        wandb.log({ 'global lr': lr, 'epoch': epoch})

        for name,data in target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                print('skipping')
                continue
            update_per_layer = weight_accumulator[name] * \
                                (1/l) * \
                                lr
            update_per_layer = torch.tensor(update_per_layer, dtype=data.dtype)

            update_per_layer = update_per_layer.cuda()
                
            if self.params['diff_privacy']:
                if 'LongTensor' in update_per_layer.type():
                    pass
                else:
                    update_per_layer.add_(self.dp_noise(data).cuda())

            data.add_(update_per_layer)

        return True

    def FedAvg_w_multiKrum(self, w, target_model):
        #num_users = num_users; num_comps = num_comps ## n, f
        num_users = self.params['partipant_sample_size']
        num_comps = self.params['number_of_adversaries']

        w_avg = copy.deepcopy(w[0])

        for k in w_avg.keys():
            w_avg[k] = torch.zeros_like(w_avg[k])

        score = np.array([0.0 for i in range(len(w))])
        score_each = np.array([[0.0 for j in range(len(w))] for i in range(len(w))])

        tmp = copy.deepcopy(w)

        ### Make flat mat for all w[i] ###
        flat_mat = [[] for i in range(len(w))]

        for k in w_avg.keys():
            for i in range(len(w)):
                A = tmp[i][k].cpu().numpy()
                flat_A = A.flatten()
                flat_mat[i] = np.concatenate((flat_mat[i], flat_A), axis=0)

        #### l2-norm(A, B) ####
        for i in range(len(tmp)):
            for j in range(len(tmp)):
                if i == j:
                    continue
                else:
                    norm_diff = np.linalg.norm(flat_mat[i] - flat_mat[j], ord=2)
                    norm_diff = norm_diff ** 2
                score_each[i][j] += norm_diff;

        #tmp = copy.deepcopy(w)
        #difference_tmp = copy.deepcopy(w)

        ## l = n - f - 2
        l = num_users - num_comps - 2

        ## To extract l nearest vectors
        for i in range(len(w)):
            score_each[i] = np.sort(score_each[i])
            for j in range(l):
                score[i] += score_each[i][j+1]

        sorted_idx_score = np.argsort(score)

        print()
        print('This is Multi_Krum line')
        print()

        print(sorted_idx_score)
        print()
        print(np.sort(score))

        clipped_sorted_idx_score = sorted_idx_score[:l]
        outlier_idx_score = sorted_idx_score[l:]
        print('clipped : ', clipped_sorted_idx_score)
        print('outlier_idx_score : ', outlier_idx_score)
        print('Multi-Krum finish')
        return clipped_sorted_idx_score, outlier_idx_score, l

    def FedAvg_FLAME(self, w, target_model, helper):
        
        weight_accumulator = dict()
        for name, data in target_model.state_dict().items():
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name] = torch.zeros_like(data)

        print()
        print()
        print("This is FLAME line")
        print()
        print()

        num_users = self.params['partipant_sample_size']
        num_comps = self.params['number_of_adversaries']
        print('num_users : ', num_users)
        print('num_comps : ', num_comps)

        w_glob = target_model.state_dict()

        weight = copy.deepcopy(w)

        #Cosine distance for local weights
        score = CosSim(weight)
        score = 1. - score

        weight = copy.deepcopy(w)
        weight_glob = copy.deepcopy(w_glob)

        ### refer to Backdoor Critical paper in https://github.com/zhmzm/Poisoning_Backdoor-critical_Layers_Attack/blob/main/utils/defense.py
        clustering = hdbscan.HDBSCAN(min_cluster_size=int((num_users/2) + 1), min_samples=1, allow_single_cluster=True).fit(score)

        labels = clustering.labels_

        print(np.shape(score), score)

        # get cluster label and Outlier label // filtering by HDBSCAN
        cluster_labels = clustering.labels_
        #outlier_scores_ = clustering.outlier_scores_

        print("Results are :")
        print(cluster_labels)
        #print(outlier_scores_)

        # Get the indices of outliers and inliers based on cluster_labels
        outlier_indices = np.where(cluster_labels == -1)[0]
        inlier_indices = np.where(cluster_labels != -1)[0]

        cnt = 0
        for i in outlier_indices:
            del weight[i-cnt]
            cnt += 1

        # for calculating l2-norm
        difference_tmp = copy.deepcopy(weight)

        #weight euclidian distance and S_t // clipping
        total_norm = [0. for i in range(len(weight))]

        #### Make flat matrix for all A_i and B_i
        ## First, to compute A_i - B_i ##
        #flat_mat = [np.array([], dtype=np.float32) for i in range(len(weight))]
        flat_mat = [[] for i in range(len(weight))]

        for k in difference_tmp[0].keys():
            for i in range(len(weight)):
                w_tmp = weight[i][k].cpu().numpy()
                w_glob_tmp = weight_glob[k].cpu().numpy()
                difference_tmp[i][k] = w_glob_tmp - w_tmp
                flat_dif_tmp = difference_tmp[i][k].flatten()
                flat_mat[i] = np.concatenate((flat_mat[i], flat_dif_tmp), axis=0)

        print(type(flat_mat), np.shape(flat_mat), flat_mat)

        ## Second, compute l2-norm ##
        for i in range(len(weight)):
            total_norm[i] = np.linalg.norm(flat_mat[i], ord=2)

        print("Total norm is :")
        print(total_norm)
        print()

        #norm_tmp = [total_norm[idx] for idx in inlier_indices]
        S_t = np.median(total_norm)
        gamma = [0. for i in range(len(weight))]

        for idx in range(len(weight)):
            #print(norm_tmp[idx])
            gamma[idx] = S_t/total_norm[idx]

        print("gamma is : ")
        print(gamma)

        for k in difference_tmp[0].keys():
            for idx in range(len(weight)):
                #weight[idx][k] = weight[idx][k]
                weight[idx][k] = weight_glob[k] + (weight[idx][k] - weight_glob[k])*min(1,gamma[idx])

        #noising // Adaptive Noising
        #noise_eps = 10000 # privacy tolerance - default 0.1
        #noise_delta = 0.05 # tnoise distribution control paramete
        #noise_lambda = (1/noise_eps)* math.sqrt(2 * math.log(1.25/noise_delta) )
        #noise_level = S_t*noise_lambda
        # lambda is decided in B.3 in appendix in this paper
        noise_lambda = 0.001
        noise_level = noise_lambda * S_t

        print('noise_level : ', noise_level)

        # Init for gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ## compute global weight
        w_avg = copy.deepcopy(weight[0])
        l = len(weight)
        print('l : ', l)
        for k in w_avg.keys():
            for i in range(1, l):
                if w_avg[k].dtype == weight[i][k].dtype:
                    w_avg[k] += weight[i][k]
                else:
                    tmp = weight[i][k].clone().detach()
                    tmp = tmp.to(w_avg[k].dtype)
                    w_avg[k] += tmp

            noise_tmp = torch.normal(0.0, noise_level, size=w_avg[k].size())
            noise_tmp = noise_tmp.to(device)
            #w_avg[k] = torch.div(w_avg[k], l)
            w_avg[k] = torch.div(w_avg[k], l) + noise_tmp

        for name, data in target_model.state_dict().items():
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name].add_(w_avg[name] - data)

        for name,data in target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                print('skipping')
                continue

            update_per_layer = weight_accumulator[name]

            update_per_layer = torch.tensor(update_per_layer, dtype=data.dtype)

            update_per_layer = update_per_layer.cuda()

            data.add_(update_per_layer)

        #self.target_model.load_state_dict(w_avg)
        #print('target_model : ', target_model)

        return True
