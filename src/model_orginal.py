#!/usr/bin/env python
# _*_ coding:utf-8 _*_

from src.Roberta import MultiHeadAttention
from transformers import AutoModel, AutoConfig, AutoTokenizer
import torch
import torch.nn as nn
from itertools import accumulate
import torch.nn as nn
from torch.nn import CrossEntropyLoss, KLDivLoss, NLLLoss, BCELoss
import torch.nn.functional as F
import torch
import numpy as np
from policy.iql import iql
from policy.qmix import qmix

Batch = 1

# epsilon = 0.1  # 探索率

# 是否强制actions=7，即同时选择3个向量
# all_3_vector = False  # True  # False
# choose-actions后不更新obs
update_obs = True  # True  # False


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    图注意力层
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):

        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征维度
        self.out_features = out_features  # 节点表示向量的输出特征维度
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活

        # 定义可训练参数，即论文中的W和a 增加Batch部分
        self.W = nn.Parameter(torch.zeros(size=(Batch, in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # xavier初始化

        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        # 输入sequence_output1 [4,373,1024]     , span_matrix [4,373,373]
        # 输出[4,373,1024]
        # 本来是二维、二维，现在多了个batch
        """
        inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵 维度[N, N] 非零即一，数据结构基本知识
        """

        # h = torch.mm(inp, self.W)# W可训练参数，初始化为torch.zeros(size=(in_features, out_features))=(1024,1024)
        # print(63,inp.shape, self.W.shape)
        h = torch.bmm(inp, self.W)  # 现在改为有batch的乘法:
        # [4,373,1024]*[4,1024,1024]->[4,373,1024]  # [N, out_features] 矩阵乘法

        N = h.size()[1]  # [0]  # N 图的节点数=373

        # a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # 本来[N, out_features]->[N, N, 2*out_features]
        a_input = torch.cat([h.repeat(1, 1, N).view(Batch, N * N, -1), h.repeat(1, N, 1)], dim=2).view(Batch, N, -1,
                                                                                                       2 * self.out_features)
        # print(59,a_input.shape)# [Batch,N, N, 2*out_features]   torch.Size([4, 287, 287, 2048])
        # a = [Batch,2 * out_features, 1] = [4,1024*2,1] 应该不用batch
        # 本来是[N, N, 2*out_features]、[1024*2,1]  现在[B,n,n,2*1024]、[1024*2,1]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # 没有强制规定维度和大小，可以用利用广播机制进行不同维度的相乘操作
        # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）
        # print(65,e.shape)#[B,N,N]=[4,373,373]

        zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷
        # print(e.device,adj.device)# cuda cpu
        attention = torch.where(adj > 0, e, zero_vec)  # [N, N] adj > 0则e
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=2)  # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [4,373,373]、[4,373,1024]->[4,373,1024]
        # [N, N].[N, out_features] => [N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    # GAT(config.hidden_size=1024, config.hidden_size=1024, 256, 0.5, 1, 3)
    def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads):
        """Dense version of GAT
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        # 定义multi-head的图注意力层            (hidden_size=1024,hidden_size=1024,0.5,1,)
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        #                                  (1024*3, 256, 0.5,   1,False)
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)

    # 再这部分加上batch即可
    def forward(self, x, adj):  # 输入sequence_output1 [4,373,1024]     , span_matrix [4,373,373]
        # x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)  # 将每个head得到的[4,373,1024]表示进行拼接
        # print(109,"allocated:{}".format(torch.cuda.memory_allocated(0)))
        # print(108,x.shape)# 应该是[4,373,1024*3]
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        # print(112,"allocated:{}".format(torch.cuda.memory_allocated(0)))
        x = F.elu(self.out_att(x, adj))  # [4,373,1024*3], span_matrix[4,373,373]输出[4,373,1024*3]并激活
        # print(114,"allocated:{}".format(torch.cuda.memory_allocated(0)))
        return F.log_softmax(x, dim=2)  # log_softmax速度变快，保持数值稳定


# 新建的模型 - 拼成3维
# return [4,502,502]
def get_span_matrix_4D(span_list, pad_token_label_id):
    '''
    span_list: [N,B,L] = [4,502]
    return span:[N,B,L,L]
    '''
    # [N,B,L]

    B, L = span_list.shape  # 4 502

    span = get_span_matrix_3D(span_list, pad_token_label_id)

    # matrix = span.contiguous().view(L, L)# .contiguous()会强制拷贝一份tensor,但是两个tensor完全没有联系
    matrix = span.contiguous().view(B, L, L)  # .contiguous()会强制拷贝一份tensor,但是两个tensor完全没有联系

    return matrix


def make_src_mask(src, src_pad_idx):
    src_mask = (src != src_pad_idx)

    return src_mask


def get_span_matrix_3D(span_list, pad_token_label_id):
    N, L = span_list.shape  # 4 502
    # print(123,span_list.shape)# torch.Size([4, 502])
    # print(124,span_list.unsqueeze(dim=-1).shape)# torch.Size([4, 502,1])
    span = span_list.unsqueeze(dim=-1).repeat(1, 1, L)
    # print(126, span.shape)# torch.Size([4, 502,502])

    mask = make_src_mask(span, pad_token_label_id)  # 应该是根据padding来构造mask

    matrix = (span.transpose(-1, -2) == span).float()

    final = torch.mul(mask.float(), matrix)  # 矩阵点乘 应该是应用mask

    return final


class Span_Classifier_Dep_New(nn.Module):  # config
    def __init__(self, config):
        # 传入那边的config，也有device + config.hidden_dropout_prob + config.hidden_size
        super().__init__()  # config

        self.device_ = config.device  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.span_num_labels = span_num_labels
        # self.bert = BertModel(config)
        # print(141,config.hidden_dropout_prob) 0.1
        self.dropout = nn.Dropout(0.1)  # config.hidden_dropout_prob=0.1 ok .hidden_dropout_prob
        self.hidden_size = config.policy_hidden_size  # 同iql的hidden_size 1536
        # print(179,self.hidden_size) 1536

        # self.graph_encoder = GAT(config.hidden_size, config.hidden_size, 256, 0.5, 1, 3)
        self.graph_encoder = GAT(config.bert_hidden_size, self.hidden_size, self.hidden_size, 0.5, 1, 3)

        # self.span = nn.Parameter(torch.randn(type_num_labels, config.hidden_size, span_num_labels))
        # self.classifier_bio = nn.Linear(config.hidden_size, span_num_labels)

        self.dense = nn.Linear(config.bert_hidden_size, self.hidden_size)  # GAT代码是256,转换维度+效果好 设置为1536同bert的hidden size

        # self.init_weights()# 报错没有这个属性？ 应该是bert里面才有这个吧

    # 返回3个向量
    def forward(
            self, outputs,  # [4, 384/..., 1024]
            pad_token_label_id=-100,
            con_spans=None,  # add
            dep=None
    ):
        assert con_spans != None
        assert dep != None

        # 直接传入outputs
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids,
        #     position_ids=position_ids,
        #     head_mask=head_mask,
        #     inputs_embeds=inputs_embeds,
        #     output_hidden_states=True,
        # )

        # final_embedding = outputs # outputs[0]  # B, L, D

        # graph_out_span
        sequence_output1 = self.dropout(outputs)  # 需要调参的地方，需要调整概率

        bert_out2 = self.dense(sequence_output1)  # 全连接层之前加入后效果较好 应该是指 降维为256

        # sequence_output2 = sequence_output2.squeeze()
        # dep = dep.contiguous().view(dep.shape[1], dep.shape[1])
        # print(212,"allocated:{}".format(torch.cuda.memory_allocated(0)))

        graph_out_t = self.graph_encoder(sequence_output1, dep)

        # 把这个代码复制到另一边作为第三维。另一边同理，这样实现互相影响
        # graph_out_t = graph_out_t.unsqueeze(0)
        # print(216,"allocated:{}".format(torch.cuda.memory_allocated(0)))

        # print(197,"allocated:{}".format(torch.cuda.memory_allocated(0)))

        span_matrix = get_span_matrix_4D(con_spans, pad_token_label_id)  # 得到邻接矩阵 [4句话,502,502]

        # print(180,span_matrix.shape)# [4,502,502]
        # sequence_output1 = sequence_output1.squeeze()
        # print(182,sequence_output1.shape)# [4，373，1024]
        # print(202,"allocated:{}".format(torch.cuda.memory_allocated(0)))# GAT耗费大内存！

        # print('sequence_output1.shape', sequence_output1.shape)  # torch.Size([4, 287, 1024])

        graph_out_s = self.graph_encoder(sequence_output1, span_matrix)  # 其实是GAT网络，得到graph的输出

        # print(203,graph_out_s.shape) torch.Size([4, 246, 256])
        # graph_out_s = graph_out_s.unsqueeze(0)  # unsqueeze 起升维的作用,参数表示在哪个地方加一个维度
        # print(206,"allocated:{}".format(torch.cuda.memory_allocated(0)))
        # graph_out_type

        return bert_out2, graph_out_t, graph_out_s

class BertWordPair(nn.Module):  # config由main读取yaml文件传入，影响iql+span_dep_model
    def __init__(self, config):
        super(BertWordPair, self).__init__()
        print("update_obs:", update_obs)
        self.mode = "train"
        self.policy_type = config.policy_type

        self.bert = AutoModel.from_pretrained(config.bert_path)

        bert_config = AutoConfig.from_pretrained(config.bert_path)

        # 对齐hidden_size 即sequence_outputs的[,,256*3]
        self.policy_hidden_size = config.policy_hidden_size  # bert_config.hidden_size  # config中 1536
        # print(496,self.policy_hidden_size)# 1536
        # config.policy_hidden_size = bert_config.hidden_size
        self.inputs_shape = config.policy_hidden_size    # 网络的输入shape  attention的维度问题？
        if self.policy_type == "iql":
            self.policy = iql(config)
            print("iql policy:", config.policy_type)
        elif self.policy_type == "qmix":
            self.policy = qmix(config)
            print("qmix policy:", config.policy_type)
        else:
            raise NameError

        bert_config.device = config.device
        self.device = config.device
        config.bert_hidden_size = bert_config.hidden_size  # 1536
        self.span_dep_model = Span_Classifier_Dep_New(config)  # 生成3个向量
        # print(17,bert_cotoken_numnfig.hidden_size) # 1024

        self.inner_dim = 256

        self.dense0 = nn.Linear(self.inputs_shape, self.inner_dim * 4 * 6)
        self.dense1 = nn.Linear(self.inputs_shape, self.inner_dim * 4 * 3)
        self.dense2 = nn.Linear(self.inputs_shape, self.inner_dim * 4 * 4)

        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)  # 0.1

        att_head_size = int(bert_config.hidden_size / bert_config.num_attention_heads)

        self.reply_attention = MultiHeadAttention(bert_config.num_attention_heads, self.inputs_shape,
                                                  att_head_size, att_head_size,
                                                  bert_config.attention_probs_dropout_prob)
        self.speaker_attention = MultiHeadAttention(bert_config.num_attention_heads, self.inputs_shape,
                                                    att_head_size, att_head_size,
                                                    bert_config.attention_probs_dropout_prob)
        self.thread_attention = MultiHeadAttention(bert_config.num_attention_heads, self.inputs_shape,
                                                   att_head_size, att_head_size,
                                                   bert_config.attention_probs_dropout_prob)

        self.config = config

        self.actions_con = [5 for _ in range(1000)]# graph_out_s = con_spans
        self.actions_dep = [6 for _ in range(1000)]# graph_out_t = dep
        self.actions_con_dep = [7 for _ in range(1000)]

        # TODO: 添加roberta-large
        if self.config.bert_path.split("/")[-1] in ['deberta-v3-base', 'deberta-v2-xlarge', 'xlm-roberta-large',
                                                    'xlm-roberta-base', 'roberta-large', 'deberta-v3-base-absa-v1.1']:
            self.tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
            special_tokens_dict = {'additional_special_tokens': ['[start]', '[end]']}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
            self.bert.resize_token_embeddings(len(self.tokenizer))

    def custom_sinusoidal_position_embedding(self, token_index, pos_type):
        """
        See RoPE paper: https://arxiv.org/abs/2104.09864
        """
        output_dim = self.inner_dim
        position_ids = token_index.unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float).to(self.config.device)
        if pos_type == 0:
            indices = torch.pow(10000, -2 * indices / output_dim)
        else:
            indices = torch.pow(15, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((1, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (1, len(token_index), output_dim))
        embeddings = embeddings.squeeze(0)
        return embeddings

    def get_instance_embedding(self, qw: torch.Tensor, kw: torch.Tensor, token_index, thread_length, pos_type):
        """_summary_
        Parameters
        ----------
        qw : torch.Tensor, (seq_len, class_nums, hidden_size)
        kw : torch.Tensor, (seq_len, class_nums, hidden_size)
        """

        seq_len, num_classes = qw.shape[:2]

        accu_index = [0] + list(accumulate(thread_length))

        logits = qw.new_zeros([seq_len, seq_len, num_classes])

        for i in range(len(thread_length)):
            for j in range(len(thread_length)):
                rstart, rend = accu_index[i], accu_index[i + 1]
                cstart, cend = accu_index[j], accu_index[j + 1]

                cur_qw, cur_kw = qw[rstart:rend], kw[cstart:cend]
                x, y = token_index[rstart:rend], token_index[cstart:cend]

                # This is used to compute relative distance, see the matrix in Fig.8 of our paper
                x = - x if i > 0 and i < j else x
                y = - y if j > 0 and i > j else y

                x_pos_emb = self.custom_sinusoidal_position_embedding(x, pos_type)
                y_pos_emb = self.custom_sinusoidal_position_embedding(y, pos_type)

                # Refer to https://kexue.fm/archives/8265
                x_cos_pos = x_pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
                x_sin_pos = x_pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
                cur_qw2 = torch.stack([-cur_qw[..., 1::2], cur_qw[..., ::2]], -1)
                cur_qw2 = cur_qw2.reshape(cur_qw.shape)
                cur_qw = cur_qw * x_cos_pos + cur_qw2 * x_sin_pos

                y_cos_pos = y_pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
                y_sin_pos = y_pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
                cur_kw2 = torch.stack([-cur_kw[..., 1::2], cur_kw[..., ::2]], -1)
                cur_kw2 = cur_kw2.reshape(cur_kw.shape)
                cur_kw = cur_kw * y_cos_pos + cur_kw2 * y_sin_pos

                pred_logits = torch.einsum('mhd,nhd->mnh', cur_qw, cur_kw).contiguous()
                logits[rstart:rend, cstart:cend] = pred_logits

        return logits

    def get_ro_embedding(self, qw, kw, token_index, thread_lengths, pos_type):
        # qw_res = qw.new_zeros(*qw.shape)
        # kw_res = kw.new_zeros(*kw.shape)
        logits = []
        batch_size = qw.shape[0]
        for i in range(batch_size):
            pred_logits = self.get_instance_embedding(qw[i], kw[i], token_index[i], thread_lengths[i], pos_type)
            logits.append(pred_logits)
        logits = torch.stack(logits)
        return logits

    def classify_matrix(self, kwargs, sequence_outputs, input_labels, masks, mat_name='ent'):

        utterance_index, token_index, thread_lengths = [kwargs[w] for w in
                                                        ['utterance_index', 'token_index', 'thread_lengths']]
        if mat_name == 'ent':
            outputs = self.dense0(sequence_outputs)
        elif mat_name == 'rel':
            outputs = self.dense1(sequence_outputs)
        else:
            outputs = self.dense2(sequence_outputs)

        outputs = torch.split(outputs, self.inner_dim * 4, dim=-1)

        outputs = torch.stack(outputs, dim=-2)

        q_token, q_utterance, k_token, k_utterance = torch.split(outputs, self.inner_dim, dim=-1)

        if self.config.use_rope == True:
            if mat_name == 'ent':
                pred_logits = self.get_ro_embedding(q_token, k_token, token_index, thread_lengths,
                                                    pos_type=0)  # pos_type=0 for token-level relative distance encoding
            else:
                pred_logits0 = self.get_ro_embedding(q_token, k_token, token_index, thread_lengths, pos_type=0)
                pred_logits1 = self.get_ro_embedding(q_utterance, k_utterance, utterance_index, thread_lengths,
                                                     pos_type=1)  # pos_type=1 for utterance-level relative distance encoding
                pred_logits = pred_logits0 + pred_logits1
        else:
            # without rope, use dot-product attention directly
            pred_logits = torch.einsum('bmhd,bnhd->bmnh', q_token, k_token).contiguous()

        nums = pred_logits.shape[-1]

        criterion = nn.CrossEntropyLoss(
            sequence_outputs.new_tensor([1.0] + [self.config.loss_weight[mat_name]] * (nums - 1)))

        active_loss = masks.view(-1) == 1
        active_logits = pred_logits.view(-1, pred_logits.shape[-1])[active_loss]
        active_labels = input_labels.view(-1)[active_loss]
        loss = criterion(active_logits, active_labels)

        return loss, pred_logits

    def build_attention(self, sequence_outputs, speaker_masks=None, reply_masks=None, thread_masks=None):
        """
        sequence_outputs: batch_size, seq_len, hidden_size
        speaker_matrix: batch_size, num, num
        head_matrix: batch_size, num, num
        """
        speaker_masks = speaker_masks.bool().unsqueeze(1)
        reply_masks = reply_masks.bool().unsqueeze(1)
        thread_masks = thread_masks.bool().unsqueeze(1)

        rep = self.reply_attention(sequence_outputs, sequence_outputs, sequence_outputs, reply_masks)[0]
        thr = self.thread_attention(sequence_outputs, sequence_outputs, sequence_outputs, thread_masks)[0]
        sp = self.speaker_attention(sequence_outputs, sequence_outputs, sequence_outputs, speaker_masks)[0]

        r = torch.stack((rep, thr, sp), 0)
        r = torch.max(r, 0)[0]
        return r

    def merge_sentence(self, sequence_outputs, input_masks, dialogue_length):
        res = []
        ends = list(accumulate(dialogue_length))
        starts = [w - z for w, z in zip(ends, dialogue_length)]
        for i, (s, e) in enumerate(zip(starts, ends)):
            stack = []
            for j in range(s, e):
                lens = input_masks[j].sum()
                stack.append(sequence_outputs[j, :lens])
            res.append(torch.cat(stack))
        new_res = sequence_outputs.new_zeros([len(res), max(map(len, res)), sequence_outputs.shape[-1]])
        for i, w in enumerate(res):
            new_res[i, :len(w)] = w
        return new_res

    def forward(self, mode, epsilon, **kwargs):

        self.mode = mode
        input_ids, input_masks, input_segments = [kwargs[w] for w in ['input_ids', 'input_masks', 'input_segments']]
        ent_matrix, rel_matrix, pol_matrix = [kwargs[w] for w in ['ent_matrix', 'rel_matrix', 'pol_matrix']]
        reply_masks, speaker_masks, thread_masks = [kwargs[w] for w in ['reply_masks', 'speaker_masks', 'thread_masks']]
        sentence_masks, full_masks, dialogue_length = [kwargs[w] for w in
                                                       ['sentence_masks', 'full_masks', 'dialogue_length']]
        multi_input_ids, multi_input_masks, multi_input_segments = [kwargs[w] for w in
                                                                    ['multi_input_ids', 'multi_input_masks',
                                                                     'multi_input_segments']]
        multi_dialogue_length = [kwargs[w] for w in ['multi_dialogue_length']]

        con_spans, dep = [kwargs[w] for w in ['con', 'dep']]  # ,转为[N,B,L]=[4,,,]

        sequence_outputs = self.bert(input_ids, token_type_ids=input_segments, attention_mask=input_masks)[0]
        # multi_sequence_outputs = self.bert(multi_input_ids, token_type_ids=multi_input_segments, attention_mask=multi_input_masks)[0]
        # multi_sequence_outputs = self.merge_sentence(multi_sequence_outputs, multi_input_masks, multi_dialogue_length[0])

        _, length_node, _ = sequence_outputs.shape

        # print("1句话token数量：", length_node)
        con_spans = con_spans[:, 0:length_node].to(self.device)
        # print(740,con_spans)
        dep = dep[:, 0:length_node, 0:length_node].to(self.device)

        bert_out, graph_out_t, graph_out_s = self.span_dep_model(sequence_outputs, pad_token_label_id=-100,
                                                                 con_spans=con_spans, dep=dep)

        # print(717,bert_out.shape) #714 torch.Size([1, 249, 1024])
        # torch.Size([502, 2, 8]) torch.Size([447, 2, 8])
        # print(750,last_epoch_actions.shape,last_epoch_actions[0:length_node,:,:].shape)
        episode = dict(  # obs根据之前的epoch的actions选择3个向量
            obs = bert_out+graph_out_t+graph_out_s,# [1，373，1024]  # 3*[1，373，1024]->[1，373，1024*3]
            #                               (episode['token_num'],2,1)
            # obs=self.policy.get_sequence_output(last_epoch_actions[0:length_node, :, :], bert_out, graph_out_t,graph_out_s, length_node),
            token_num=length_node,  # 表示这句话的token数量
            device=self.device
        )
        # print(728,episode['obs'].shape) # torch.Size([1, 318, 3072])
        if self.config.ablation==5:# 5-con/6-dep/7-con+dep graph_out_s = con_spans
            actions = self.actions_con[0:length_node]
            # print(547,actions)
        elif self.config.ablation==6:# graph_out_t = dep
            actions = self.actions_dep[0:length_node]
            # print(547, actions)
        elif self.config.ablation==7:
            actions = self.actions_con_dep[0:length_node]
            # print(547, actions)
        else:
            actions = self.policy.choose_actions(episode, epsilon=epsilon,mode=mode)
        # actions[0] = torch.tensor(actions[0]).unsqueeze(0).reshape(episode['token_num'], 1)
        # actions[1] = torch.tensor(actions[1]).unsqueeze(0).reshape(episode['token_num'], 1)
        # actions = torch.cat(actions, dim=1)  # (episode['token_num'],2)
        # actions = actions.unsqueeze(-1).to(self.device)  # ->(episode['token_num'],2,1)
        # if self.mode != "train":  # eval时候看actions 看的全都choose 1
        #     print(766, actions[:, 0, 0])  # (episode['token_num'],2,1)
        sequence_outputs = self.policy.get_sequence_output(actions, bert_out, graph_out_t, graph_out_s,
                                                           episode['token_num'])
        # print(737, "allocated:{}".format(torch.cuda.memory_allocated(0)))
        # 根据actions更新obs(调参点)
        if update_obs:
            episode_next = dict(  # obs根据之前的epoch的actions选择3个向量；obs初始一直是全选！
                obs=sequence_outputs.contiguous(),
                token_num=length_node,  # 表示这句话的token数量
                device=self.device
            )
            # episode['obs'] = sequence_outputs
            # episode_next = episode # 更新obs
            # episode_next['obs'] = sequence_outputs

        # sequence_outputs =  self.merge_sentence(sequence_outputs, input_masks, dialogue_length)
        # sequence_outputs = sequence_outputs + multi_sequence_outputs

        # 减少网络深度，不然内存不够了
        if self.config.use_attention:
            sequence_outputs = self.dropout(sequence_outputs)
            sequence_outputs = self.build_attention(sequence_outputs, reply_masks=reply_masks, speaker_masks=speaker_masks,thread_masks=thread_masks)

        loss0, tags0 = self.classify_matrix(kwargs, sequence_outputs, ent_matrix, sentence_masks, 'ent')
        loss1, tags1 = self.classify_matrix(kwargs, sequence_outputs, rel_matrix, full_masks, 'rel')
        loss2, tags2 = self.classify_matrix(kwargs, sequence_outputs, pol_matrix, full_masks, 'pol')

        # 设置超参数来调整r的计算方式
        # af0, af1, af2 = 0.3, 0.4, 0.3  # 以loss来计算r
        # r = (af0 * (1 - loss0) + af1 * (1 - loss1) + af2 * (1 - loss2)) * 1
        
        r = (self.config.af0 * (1 - loss0) + self.config.af1 * (1 - loss1) + self.config.af2 * (1 - loss2)) * 1
        
        # r: 0~倍率 感觉应该r=0~1/0.5，因为q-value max是0.5。但是有可能是因为r才使q-value max=0.5

        print(536, "reward :", r.item())  # 应当逐渐递增 √

        policy_loss = self.policy.learn(episode, episode_next, actions, r)
        print("policy loss:", policy_loss.item() )

        return (loss0, loss1, loss2, policy_loss), (tags0, tags1, tags2), actions
        
        # r = r.unsqueeze(0)  # (1,1)
        # r = r.repeat(episode['token_num'], 1)  # .to(self.device)# (128,2)
        #
        # inputs_RNN = self.policy.get_all_inputs(episode)  # tensor (128,2,778)
        # # print(764,inputs_RNN.shape)# torch.Size([192, 2, 3082])
        # # inputs_RNN_next = inputs_RNN[1:]  # 不取第一个 tensor (127,2,778) 778=3*256(768)+8+2
        # # hhh = [inputs_RNN_next, torch.zeros((1, 2, self.policy_hidden_size * 3+10)).to(self.device)]
        # # inputs_RNN_next = torch.cat(hhh, dim=0)  # -> tensor (128,2,778)
        # inputs_RNN_next = self.policy.get_all_inputs(episode_next)
        #
        # actions = torch.tensor(actions).unsqueeze(0).reshape(episode['token_num'], 1, 1).to(self.device)
        # q_evals, self.policy.eval_hidden = self.policy.eval_rnn(inputs_RNN,
        #                                                         self.policy.eval_hidden)  # 输出 (128,2,hidden_size = 8)
        # self.policy.eval_hidden = self.policy.eval_hidden.data  # 验证一下生效了没
        # q_targets, self.policy.target_hidden = self.policy.target_rnn(inputs_RNN_next, self.policy.target_hidden)
        # self.policy.target_hidden = self.policy.target_hidden.data
        #
        # # q_value tensor([[-0.5730,  0.4875,  0.4863,  0.1687, -0.0487, -0.3823,  0.1119,  0.2697],
        # q_evals = torch.gather(q_evals, dim=2, index=actions).squeeze(2)  # (128,2,8) -> (128,2)
        # # print(986, q_targets.shape)# 都正常 (128,1)
        # q_targets = q_targets.max(dim=2)[0]  # [0]是因为返回的tuple  (128,2,8) -> (128,2)
        # # print(988, q_targets.shape)
        # targets = r + 0.99 * q_targets  # gama=0.99同MARL代码
        # td_error = (q_evals - targets.detach())  # 让q_evals[actions]逼近tragets
        # policy_loss = (td_error ** 2).sum() / episode['token_num']  # Mse损失函数的表达式
        #
        # # print("iql loss:", policy_loss)
        #
        # return (loss0, loss1, loss2, policy_loss), (tags0, tags1, tags2), actions
        # actions: (episode['token_num'],2,1)