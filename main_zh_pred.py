#!/usr/bin/env python


import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import yaml
import argparse

import torch
import sys
import numpy as np

from attrdict import AttrDict
from torch.optim import AdamW
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

from src.utils import MyDataLoader, RelationMetric
from src.model_orginal import BertWordPair
from src.common import set_seed, ScoreManager, update_config
from tqdm import tqdm
from loguru import logger
from tools.common import init_logger, logger
from datetime import datetime


class Main:
    def __init__(self, args):
        config = AttrDict(
            yaml.load(open('src/config_erlangshen_xlarge.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader))

        for k, v in vars(args).items():
            setattr(config, k, v)

        config = update_config(config)

        set_seed(config.seed)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        config.timestamp = timestamp
        config.target_dir += timestamp
        save_name = str(config.seed) + "-" + str(config.policy_seed)
        save_name += "-" + str(config.bert_lr) + "-" + str(config.learning_rate_policy) + "-" + str(
            config.policy_type) + "-" + str(config.use_softmax)
        save_name += "-" + str(config.max_tgt_num_spans) + "-" + str(config.epoch_size) + "-" + str(
            config.reward_weights)
        save_name += "-" + str(config.af0)+"-" + str(config.af1)+"-" + str(config.af2)
        config.target_dir += "-" + save_name
        if not os.path.exists(config.target_dir):
            os.makedirs(config.target_dir)
        init_logger(
            log_file=config.target_dir + '/model-{}_seed-{}_epoch-{}_bs-{}_lr-{}_bert_lr-{}_use_rope-{}.log'.format(
                config.bert_path.split('/')[-1], config.seed, config.epoch_size, config.batch_size, config.lr,
                config.bert_lr, config.use_rope))

        config.device = torch.device('cuda:{}'.format(config.cuda_index) if torch.cuda.is_available() else 'cpu')
        self.config = config
        # for k, v in config.items():
        #     # print('{}: {}'.format(k, v))
        #     logger.info('{}: {}'.format(k, v))
        for k, v in vars(args).items():
            logger.info('{}: {}'.format(k, v))


    def train_iter(self):
        self.model.train()
        train_data = tqdm(self.trainLoader, total=self.trainLoader.__len__(), file=sys.stdout)
        losses = []
        for i, data in enumerate(train_data):
            loss, _ = self.model(**data)
            losses.append([w.tolist() for w in loss])
            sum(loss).backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()

            description = "Epoch {}, entity loss:{:.4f}, rel loss: {:.4f}, pol loss: {:.4f}".format(self.global_epoch,
                                                                                                    *np.mean(losses, 0))
            train_data.set_description(description)
            # if i == 10:
            #     break

    def evaluate_iter(self, dataLoader=None):
        epsilon = 0.0  # 不进行探索而选择最大的q值
        self.model.eval()
        self.model.policy.eval_rnn.eval()
        self.model.policy.target_rnn.eval()
        if self.config.policy_type == "qmix":
            self.model.policy.eval_qmix_net.eval()
            self.model.policy.target_qmix_net.eval()
        dataLoader = self.validLoader if dataLoader is None else dataLoader
        dataiter = tqdm(dataLoader, total=dataLoader.__len__(), file=sys.stdout)
        mode = "eval"
        for i, data in enumerate(dataiter):
            with torch.no_grad():
                _, (pred_ent_matrix, pred_rel_matrix, pred_pol_matrix),_ = self.model(mode, epsilon, **data)
                self.relation_metric.add_instance(data, pred_ent_matrix, pred_rel_matrix, pred_pol_matrix)

    def evaluate(self, epoch=0, action='eval'):
        PATH = os.path.join(self.config.target_dir, "{}_{}.pth.tar").format(self.config.lang, epoch)
        self.model.load_state_dict(torch.load(PATH, map_location=self.config.device)['model'])
        self.model.eval()
        self.model.policy.eval_rnn.eval()
        self.model.policy.target_rnn.eval()
        if self.config.policy_type == "qmix":
            self.model.policy.eval_qmix_net.eval()
            self.model.policy.target_qmix_net.eval()

        self.evaluate_iter(self.testLoader)
        action = "pred"
        result = self.relation_metric.compute('test', action)

        if action == 'eval':
            score, res = result
            logger.info("Evaluate on test set, micro-F1 score: {:.4f}%".format(score * 100))
            logger.info(res)
            # print(res)

    def inference(self, model_path):
        # PATH = os.path.join(self.config.target_dir, "{}_{}.pth.tar").format(self.config.lang, epoch)
        PATH = model_path
        self.model.load_state_dict(torch.load(PATH, map_location=self.config.device)['model'])
        self.model.eval()
        self.model.policy.eval_rnn.eval()
        self.model.policy.target_rnn.eval()
        if self.config.policy_type == "qmix":
            self.model.policy.eval_qmix_net.eval()
            self.model.policy.target_qmix_net.eval()

        self.evaluate_iter(self.testLoader)
        action = "pred"
        result = self.relation_metric.compute('test', action)# NULL

    def train(self):
        best_score, best_iter = 0, 0
        for epoch in range(self.config.epoch_size):
            self.global_epoch = epoch
            self.train_iter()
            self.evaluate_iter()

            micro_score, iden_score, res = self.relation_metric.compute()
            score = (micro_score + iden_score) / 2
            self.score_manager.add_instance(score, res)
            logger.info("Epoch {}, micro-F1 score: {:.4f}%".format(epoch, micro_score * 100))
            logger.info("Epoch {}, iden-F1 score: {:.4f}%".format(epoch, iden_score * 100))
            logger.info("Epoch {}, both-F1 score: {:.4f}%".format(epoch, score * 100))
            print(res)

            if score > best_score:
                best_score, best_iter = score, epoch

                torch.save({'epoch': epoch, 'model': self.model.cpu().state_dict(), 'best_score': best_score},
                           os.path.join(self.config.target_dir, "{}_{}.pth.tar".format(self.config.lang, best_iter)))
                self.model.to(self.config.device)
            elif epoch - best_iter > self.config.patience:
                print("Not upgrade for {} steps, early stopping...".format(self.config.patience))
                break
            self.model.to(self.config.device)

        self.best_iter = best_iter

    def load_param(self):

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0}]

        self.optimizer = AdamW(optimizer_grouped_parameters,
                               lr=float(self.config.bert_lr),
                               eps=float(self.config.adam_epsilon))
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.config.warmup_steps,
                                                         num_training_steps=self.config.epoch_size * self.trainLoader.__len__())

        # GAT作为span_dep_model在model里面优化了？？？(确认一下)
        # 设置 iql的优化器
        param_optimizer = list(self.model.policy.eval_rnn.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0}]

        self.optimizer_eval_RNN = AdamW(optimizer_grouped_parameters,
                                        lr=float(self.config.bert_lr),
                                        eps=float(self.config.adam_epsilon))
        self.scheduler_eval_RNN = get_linear_schedule_with_warmup(self.optimizer_eval_RNN,
                                                                  num_warmup_steps=self.config.warmup_steps,
                                                                  num_training_steps=self.config.epoch_size * self.trainLoader.__len__())

        if self.config.policy_type == "qmix":
            param_optimizer = list(self.model.policy.eval_qmix_net.named_parameters())
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.config.weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0}]

            self.optimizer_qmix = AdamW(optimizer_grouped_parameters,
                                   lr=float(self.config.bert_lr),
                                   eps=float(self.config.adam_epsilon))
            self.scheduler_qmix = get_linear_schedule_with_warmup(self.optimizer_qmix, num_warmup_steps=self.config.warmup_steps,
                                                             num_training_steps=self.config.epoch_size * self.trainLoader.__len__())

    def forward(self):
        self.trainLoader, self.validLoader, self.testLoader, config = MyDataLoader(self.config).getdata()
        self.model = BertWordPair(self.config).to(config.device)
        self.score_manager = ScoreManager()
        self.relation_metric = RelationMetric(self.config)
        self.load_param()

        # if self.config.action != 'train':
        #     logger.info("Start to {}...".format(self.config.action))
        #     self.evaluate(self.config.best_iter, self.config.action)
        #     return

        # logger.info("Start training...")
        # self.train()
        # logger.info("Training finished..., best epoch is {}...".format(self.best_iter))
        # if 'test' in self.config.input_files:
        #     logger.info("Start evaluating...")
        #     self.evaluate(self.best_iter)
        logger.info("Start testing...")
        # a = '/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/coco/ACL/project/data/save/zh/iql-42/'
        # model_path = a + "zh_17.pth.tar"
        model_path = self.config.path
        self.inference(model_path)


#

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-l', '--lang', type=str, default='zh', choices=['zh', 'en'], help='language selection')
    # parser.add_argument('-bl', '--bert_lr', type=float, default=1e-5, help='learning rate for BERT layers')
    # parser.add_argument('-c', '--cuda_index', type=int, default=0, help='CUDA index')

    parser.add_argument('-bert_lr', '--bert_lr', type=float, default=1e-5, help='learning rate for BERT layers')
    # parser.add_argument('-c', '--cuda_index', type=int, default=0, help='CUDA index')
    parser.add_argument('-learning_rate_policy', '--learning_rate_policy', type=float, default=1e-5,help='learning rate for iql')
    parser.add_argument('-policy_hidden_size', '--policy_hidden_size', type=int, default=1024,
                        help='hidden_size for iql')
    parser.add_argument('-use_softmax', '--use_softmax', type=int, default=0, help='use softmax or not')
    parser.add_argument('-policy_type', '--policy_type', type=str, help='choose policy for training')
    parser.add_argument('-bert_hidden_size', '--bert_hidden_size', type=int,
                        help='record bert_hidden_size')# 用于实现bert的hidden-size可以和config中不同
    parser.add_argument('-policy_seed', '--policy_seed', type=int, default=0, help='random seed')
    parser.add_argument('-max_tgt_num_spans', '--max_tgt_num_spans', type=int, default=10 )
    parser.add_argument('-reward_weights', '--reward_weights', type=int, default=1 )
    parser.add_argument('-epoch_size', '--epoch_size', type=int, default=1 )

    parser.add_argument('-af0', '--af0', type=float, default=0.3 )
    parser.add_argument('-af1', '--af1', type=float, default=0.4 )
    parser.add_argument('-af2', '--af2', type=float, default=0.3 )
    parser.add_argument('-use_attention', '--use_attention', type=int, default=0, help='use extra attention or not')

    parser.add_argument('-i', '--input_files', type=str, default='train valid test', help='input file names')
    parser.add_argument('-a', '--action', type=str, default='train', choices=['train', 'eval', 'pred'],
                        help='choose to train, evaluate, or predict')
    parser.add_argument('-b', '--best_iter', type=int, default=0,
                        help='best iter to run test, only used when action is eval or pred')
    parser.add_argument('-path', '--path', type=str, help='')
    # parser.add_argument('-s', '--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()
    if args.use_softmax:
        args.use_softmax = True
    else:
        args.use_softmax = False
    main = Main(args)
    main.forward()
