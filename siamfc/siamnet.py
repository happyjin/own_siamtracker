import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from siamfc.config import config
from siamfc.convGRU import ConvGRU
from functools import reduce
import matplotlib.pyplot as plt

class SiameseNet(nn.Module):
    def __init__(self, dtype):
        super(SiameseNet, self).__init__()
        # utilize AlexNet
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=2),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, 5),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3),
            nn.BatchNorm2d(256),
        )

        self.tar_weights = nn.Parameter(torch.empty(256,64,3,3)) # weights for target model function. how to kaiming initialization???
        nn.init.kaiming_normal_(self.tar_weights.data, mode='fan_out', nonlinearity='relu') # param. initialization

        gt, weight = self._create_gt_mask((config.response_sz, config.response_sz))
        self.gt = torch.from_numpy(gt).type(dtype)
        self.weight = torch.from_numpy(weight).type(dtype)

        self.exemplar = None
        # initialize stacked convolutional-GRU
        self.conv_gru = ConvGRU(input_size=(6,6),
                                input_dim=256,
                                hidden_dim=64,
                                kernel_size=(3,3),
                                num_layers=2,
                                dtype=dtype,
                                return_all_layers=True)

    def init_weights(self):
        """
        kaiming initialization for weights and biases
        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu') ##??? fan_in and fan_out difference?
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _create_gt_mask(self, shape):
        # same for all pairs
        h, w = shape
        y = np.arange(h, dtype=np.float32) - (h-1) / 2.
        x = np.arange(w, dtype=np.float32) - (w-1) / 2.
        y, x = np.meshgrid(y, x)
        dist = np.sqrt(x**2 + y**2)
        mask = np.zeros((h, w))
        mask[dist <= config.radius / config.total_stride] = 1
        mask = mask[np.newaxis, :, :]
        weights = np.ones_like(mask)
        weights[mask == 1] = 0.5 / np.sum(mask == 1)
        weights[mask == 0] = 0.5 / np.sum(mask == 0)
        mask = np.repeat(mask, config.train_batch_size, axis=0)[:, np.newaxis, :, :]
        return mask.astype(np.float32), weights.astype(np.float32)

    def cls_loss(self, pred):
        """
        normalized classification loss
        :param pred: (b,c,h,w)
            predicted score map
        :return: float
            sum of a batch of weighted loss value
        """
        # normalized by ((n-m+1)x(n-m+1))
        return F.binary_cross_entropy_with_logits(pred, self.gt,
                    self.weight, reduction='elementwise_mean')# / pred.size(0) # batch_size

    def anchor_loss(self, target_models):
        """
        normalized anchor loss
        :param target_models: (b,c,h,w)
            target model in the first frame and the rest predicted target model for the a batch size data
        :return: float
            normalized anchor loss value for this bach size of data. normalized by (bxcxhxw)
        """
        target_init = target_models[0]
        target_pred = target_models[1:]
        n_elems = reduce(lambda x, y: x * y, list(target_pred.size())) / target_pred.size(0) # (mxmxd)
        targets_init = target_init.repeat(target_pred.size(0),1,1,1)
        distance = torch.norm(targets_init - target_pred, 2) ** 2
        return distance / n_elems

    def total_loss(self, scores, target_models):
        """
        compute the total loss
        :param scores: tensor (b,h,w)
            a batch of score maps
        :param target_models: tensor (b,c,h,w)
            a batch of predicted target models
        :return: float
            weighted total loss by combination factor between loss_anc and loss_cls
        """
        loss_anc = self.anchor_loss(target_models)
        loss_cls = self.cls_loss(scores)
        return (1 - config.anc_factor) * loss_cls + config.anc_factor * loss_anc

    def compute_score(self, last_state_list, instance, cur_exemplar):
        """

        :param last_state_list: list len(.)=2 e.g. the 1st and 2nd h_states from stacked hidden layer
            layer_output_list[1][0]: current hidden state for the last stacked layer e.g. 2nd hidden layer for cur frame
        :param instance: tensor (b,c,h,w)
            current search image feature at time frame t+1
        :param instance: tensor (b,c,h,w)
            exemplar feature at time frame t
        :return: tensor(b,w,h), tensor(b,c,h,w)
            score map and predicted target model for searching in the next time frame
        """
        h_cur = last_state_list[1][0] # h_state from the 2nd stacked hidden layer
        padding = 1, 1  # W2=(W1−F+2P)/S+1 --> 6=(6-3+2P)/1 + 1
        # compute weights of target_model by sigmoid function for target model using
        # weights is attention mechanism by choosing proper channels and height&width part of 2D features
        target_model_weights = torch.sigmoid(F.conv2d(h_cur, self.tar_weights, padding=padding))  # (b,c,h,w)
        # compute predicted target_model for the next time frame by Hadamard product weights of tar_model with cur_exemplar feature
        # compute target_model namely the updated exemplar feature

        target_model = target_model_weights * cur_exemplar
        N, C, H, W = instance.shape  # n_batches, channels, height, width
        instance = instance.view(1, N * C, H, W) # stacked instance features to channels
        score = F.conv2d(instance, target_model, groups=N)
        return score.transpose(0, 1), target_model

    def forward(self, x):
        """

        :param x: (tensor, tensor)
            exemplar tensor and instance tensor
        :return:
        """
        exemplar, instance = x
        if exemplar is not None and instance is not None:
            # in the training process
            exemplar = self.features(exemplar) # (b,t,c,h,w)
            instance = self.features(instance) # (b,c,h,w)
            exemplar = exemplar.unsqueeze(1) # (b,t,c,h,w) for conv-GRU
            target_model = exemplar[0] # (t,c,h,w) initial target model at 1st time frame

            # input data into rolled conv-GRU
            for i in range(config.train_batch_size):
                if i == 0: # unroll and initial stacked hidden states every time_steps=config.batch_siz
                    # batch_size data input into the same conv-GRU so that the last_state_list contain 1 if only 1 time_frame
                    _, last_state_list = self.conv_gru(exemplar[i].unsqueeze(0))
                    score, target_next = self.compute_score(last_state_list, instance[i].unsqueeze(0), exemplar[i])
                else:
                    hidden_state = []
                    for i in range(len(last_state_list)):
                        hidden_state.append(last_state_list[i][0]) # [i]: h_state from 1st & 2nd stacked hidden layer
                                                                   # [0]: since list append [h] namely another list so need [0] idx to extract h
                    _, last_state_list = self.conv_gru(exemplar[i].unsqueeze(0), hidden_state)
                    score_next, target_next  = self.compute_score(last_state_list, instance[i].unsqueeze(0), exemplar[i])
                    score = torch.cat((score, score_next), dim=0)
                target_model = torch.cat((target_model, target_next))
            return score, target_model[:-1] # (b,t,h,w) & remove the last one which is not useful batch_size+1-1, since we need to compute loss and initalize the next batch data
        elif exemplar is not None and instance is None:
            # inference used
            self.exemplar = self.features(exemplar)
            self.new_video = True # new_video flag
            #self.exemplar = torch.cat([self.exemplar for _ in range(3)], dim=0)
        else:
            instance = self.features(instance)
            if self.new_video:
                # for initial stacked hidden states are zeros
                _, self.last_state_list = self.conv_gru(self.exemplar.unsqueeze(0))  # exemplar.shape=(b,t,c,h,w)
                for i in range(len(instance)): # n_pyramid instances
                    if i == 0:
                        score, target_next = self.compute_score(self.last_state_list, instance[i].unsqueeze(0), self.exemplar)
                    else:
                        score_next, target_next = self.compute_score(self.last_state_list, instance[i].unsqueeze(0), self.exemplar)
                        score = torch.cat((score, score_next), dim=0)
                self.new_video = False
            else:
                # for non initial stacked hidden states
                for k in range(len(instance)): # n_pyramid instances
                    hidden_state = []
                    for i in range(len(self.last_state_list)):
                        hidden_state.append(self.last_state_list[i][0])
                    _, self.last_state_list = self.conv_gru(self.exemplar.unsqueeze(0), hidden_state)
                    if k == 0:
                        score, target_next = self.compute_score(self.last_state_list, instance[i].unsqueeze(0), self.exemplar)
                    else:
                        score_next, target_next  = self.compute_score(self.last_state_list, instance[i].unsqueeze(0), self.exemplar)
                        score = torch.cat((score, score_next), dim=0) # concatenate n_pyramid instance score maps
            return score # (b,t,h,w) = e.g.(n_pyramid,1,h,w)