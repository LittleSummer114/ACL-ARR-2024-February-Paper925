# 修改MTD的data_utils
# -*- coding:utf-8 -*

import logging
import os
import json
import torch.nn.functional as F
import torch
# from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import numpy as np

logger = logging.getLogger(__name__)

class InputExample(object):

    """A single training/test example for token classification."""

    def __init__(self,doc_id, guid, words,
                 # labels, spans, types,
                 con_mapnodes, dep_heads, con_heads):

        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.doc_id = doc_id
        self.guid = guid
        self.words = words
        # self.labels = labels
        # self.spans = spans
        # self.types = types

        self.con_mapnodes = con_mapnodes
        self.dep_heads = dep_heads
        self.con_heads = con_heads

class InputFeatures(object):

    """A single set of features of data."""

    def __init__(self,doc_id, select_spans,adj_matrix):
        self.doc_id = doc_id
        self.select_spans = select_spans
        self.adj_matrix = adj_matrix

# load_and_cache_examples()调用其读取json文件
def read_examples_from_file(args, data_dir, mode):

    file_path = os.path.join(data_dir, "{}_con_sdp.json".format(mode))
    guid_index = 1
    examples = []

    with open(file_path, 'r') as f:

        data = json.load(f)

        for item in data:# 只有这些：
        # "doc_id"、"str_words"、"sdp_head"、"con_head"、"con_mapnode"
        #     print(63,item)# 正常的！
            words = item["str_words"]# 一段话当成一句话
            # labels_ner = item["tags_ner"]
            # labels_esi = item["tags_esi"]
            # labels_net = item["tags_net"]

            con_mapnodes = item["con_mapnode"]
            con_heads = item["con_head"]
            dep_heads = item["sdp_head"]
            doc_id = item["doc_id"]

            examples.append(InputExample(doc_id=doc_id,guid="%s-%d".format(mode, guid_index),
                                         words=words,
                                         # labels=labels_ner,
                                         # spans=labels_esi, types=labels_net,
                                         con_mapnodes=con_mapnodes,
                                         dep_heads=dep_heads, con_heads=con_heads))

            guid_index += 1

    examples_src = []
    # print(84,examples[0].words)# yes
    return examples, examples_src

def get_path_and_children_dict(heads):

    path_dict = {}
    remain_nodes = list(range(len(heads)))
    delete_nodes = []

    while len(remain_nodes) > 0:

        for idx in remain_nodes:
            # 初始状态
            if idx not in path_dict:
                path_dict[idx] = [heads[idx]]  # no self
                if heads[idx] == -1:
                    delete_nodes.append(idx)  # need delete root

            else:

                last_node = path_dict[idx][-1]
                if last_node not in remain_nodes:
                    path_dict[idx].extend(path_dict[last_node])
                    delete_nodes.append(idx)
                else:
                    path_dict[idx].append(heads[last_node])

        # remove nodes
        for del_node in delete_nodes:
            remain_nodes.remove(del_node)
        delete_nodes = []

    # children_dict
    children_dict = {}
    for x, l in path_dict.items():
        if l[0] == -1:
            continue
        if l[0] not in children_dict:
            children_dict[l[0]] = [x]
        else:
            children_dict[l[0]].append(x)

    return path_dict, children_dict

def form_layers_and_influence_range(path_dict, mapback):

    sorted_path_dict = sorted(path_dict.items(), key=lambda x: len(x[1]))
    influence_range = {cid: [idx, idx + 1] for idx, cid in enumerate(mapback)}
    layers = {}
    node2layerid = {}

    for cid, path_dict in sorted_path_dict[::-1]:

        length = len(path_dict) - 1
        if length not in layers:
            layers[length] = [cid]
            node2layerid[cid] = length
        else:
            layers[length].append(cid)
            node2layerid[cid] = length
        father_idx = path_dict[0]

        assert (father_idx not in mapback)

        if father_idx not in influence_range:
            influence_range[father_idx] = influence_range[cid][:]  # deep copy
        else:
            influence_range[father_idx][0] = min(influence_range[father_idx][0], influence_range[cid][0])
            influence_range[father_idx][1] = max(influence_range[father_idx][1], influence_range[cid][1])

    layers = sorted(layers.items(), key=lambda x: x[0])
    layers = [(cid, sorted(l)) for cid, l in layers]  # or [(cid,l.sort()) for cid,l in layers]

    return layers, influence_range, node2layerid

def form_spans(layers, influence_range, token_len, con_mapnode, special_token='[N]'):
    spans = []
    sub_len = len(special_token)

    for _, nodes in layers:

        pointer = 0
        add_pre = 0
        temp = [0] * token_len
        temp_indi = ['-'] * token_len

        for node_idx in nodes:
            begin, end = influence_range[node_idx]

            if con_mapnode[node_idx][-sub_len:] == special_token:
                temp_indi[begin:end] = [con_mapnode[node_idx][:-sub_len]] * (end - begin)

            if (begin != pointer):
                sub_pre = spans[-1][pointer]
                temp[pointer:begin] = [x + add_pre - sub_pre for x in spans[-1][pointer:begin]]  #
                add_pre = temp[begin - 1] + 1
            temp[begin:end] = [add_pre] * (end - begin)

            add_pre += 1
            pointer = end
        if pointer != token_len:
            sub_pre = spans[-1][pointer]
            temp[pointer:token_len] = [x + add_pre - sub_pre for x in spans[-1][pointer:token_len]]
            add_pre = temp[begin - 1] + 1

        spans.append(temp)

    return spans

def sdphead_to_adj_oneshot(heads, word_mapback,sent_len,
                        leaf2root=True, root2leaf=True, self_loop=True):
    """
    Convert a sequence of head indexes into a 0/1 matirx.
    """

    edge_index = []

    adj_matrix = np.zeros((sent_len, sent_len), dtype=np.float32)

    heads = heads[:sent_len]

    for i in range(len(word_mapback)):

        index = word_mapback[i]

        head = heads[index]

        for onehead in head:

            if(onehead!=i):

                if onehead != -1:

                    if leaf2root:
                        adj_matrix[onehead, i] = 1

                        edge_index.append((onehead, i))

                    if root2leaf:

                        adj_matrix[i, onehead] = 1
                        edge_index.append((i, onehead))

        if self_loop:

            adj_matrix[i, i] = 1

    return adj_matrix,edge_index
# 计算features的（利用了dep_head+con_head）
def convert_examples_to_features(max_num_spans: object,
                                 examples: object,
                                 max_seq_length: object,
                                 tokenizer: object,
                                 cls_token_at_end: object = False,# 当tokenizer为 xlnet为True
                                 cls_token: object = "[CLS]", sep_token: object = "[SEP]",
                                 sep_token_extra: object = False,# 当tokenizer为 roberta 为True
                                 pad_token_label_id: object = -100, # criterion.ignore_index也用的是-100
                                 ) -> object:

    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    features = []
    extra_long_samples = 0
    # print(442, examples[0].words) yes
    # print(442, examples[0].doc_id) ok
    for (ex_index, example) in enumerate(examples):
        # print(442, example.words) yes
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        word_mapback = []

        # print(len(example.words), len(example.labels))

        count = 0
        # print(264,examples)
        # print(264,examples[0])
        # print(examples[0].doc_id,264,examples[0].words)
        # print(examples[1].doc_id,264,examples[1].words)
        # for word, span_label, type_label in zip(example.words, example.spans, example.types):
        # for word,doc_id in zip(example.words , example.doc_id):# 为什么只有遍历4个tokens？
        # print(271,example.doc_id)
        for word in (example.words):#
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)

            if len(word_tokens) > 0:

                word_mapback.extend([count] * len(word_tokens))

            count+=1

            # full_label_ids.extend([label] * len(word_tokens))

        # print(len(tokens), len(label_ids), len(full_label_ids))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.

        con_head = example.con_heads

        con_mapnode = example.con_mapnodes

        dep_head = example.dep_heads

        con_path_dict, con_children = get_path_and_children_dict(con_head)

        sub_len = len('[N]')

        mapback = [idx for idx, word in enumerate(con_mapnode) if word[-sub_len:] != '[N]']

        layers, influence_range, node2layerid = form_layers_and_influence_range(con_path_dict, mapback)

        spans = form_spans(layers, influence_range, len(example.words), con_mapnode)

        select_spans = []

        """
        Convert a sequence of head indexes into a 0/1 matirx.
        """

        word_mapback = word_mapback[: max_seq_length]

        adj_matrix , edge_index = sdphead_to_adj_oneshot(dep_head,word_mapback,max_seq_length)

        if(len(spans)>0):

            if len(spans) <= max_num_spans:

                for word_index in word_mapback:

                    select_spans.append(spans[-1][word_index])

            else:

                temp = spans[max_num_spans]

                for word_index in word_mapback:

                    select_spans.append(temp[word_index])

        else:# 直接用pad

            select_spans = [pad_token_label_id] * max_seq_length

        special_tokens_count = 3 if sep_token_extra else 2

        select_spans = select_spans[: (max_seq_length - special_tokens_count)]

        con_edge_index = []

        for i in select_spans:
            for j in select_spans:

                if(i==j):
                    con_edge_index.append([i,j])

        if len(tokens) > max_seq_length - special_tokens_count:

            tokens = tokens[: (max_seq_length - special_tokens_count)]

            extra_long_samples += 1

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]

        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0

        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        tokens += [sep_token]
        select_spans +=  [pad_token_label_id]

        if sep_token_extra:

            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            select_spans += [pad_token_label_id]

        if cls_token_at_end:
            tokens += [cls_token]

        else:
            tokens = [cls_token] + tokens
            # 这里其实对于word_mapback的默认设置是0真的OK吗
            select_spans = [pad_token_label_id] + select_spans

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        # print(394,padding_length,tokens)# 51???
        # if pad_on_left: 我看这个参数一直都false所以删了直接用的else试试；如果报错那说明这个参数还是有用到的
        #     input_ids = ([pad_token] * padding_length) + input_ids
        #     input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        #     segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        #
        # else:
        #     input_ids += [pad_token] * padding_length
        #     input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        #
        #     select_spans+= [pad_token_label_id] * padding_length

        select_spans+= [pad_token_label_id] * padding_length

        assert len(select_spans)==max_seq_length
        assert len(adj_matrix[0]) == max_seq_length
        assert len(adj_matrix[1]) == max_seq_length

        features.append(
            InputFeatures(doc_id = example.doc_id,select_spans=select_spans,adj_matrix=adj_matrix)
        )
    return features

def load_and_cache_examples(args, tokenizer,  mode):

    # tags_to_id = tag_to_id(args.data_dir, args.dataset)

    if args.local_rank not in [-1, 0] and not evaluate:# 多机多卡模式
        torch.distributed.barrier()
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    # args.max_tgt_num_spans = 13 # 3 # MTD default=3 config里面help = type loss eps
    cached_features_file = os.path.join(
        # args.data_dir,
        "/nfs/home/yangliu/code/nlp/jn/secondary/project-DiaASQ/project/data/dataset/jsons_en/",# 1、zh这里要改
        str(mode)+ str(args.max_tgt_num_spans)+".pt",
    )
    # 直接读features
    if os.path.exists(cached_features_file):

        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    # if False:
    #     print("lll")
    else:

        max_num_spans = args.max_tgt_num_spans # default=3 config里面help = type loss eps

        logger.info("Creating features from dataset file at %s", args.data_dir)
        # 可直接用的读取文件数据函数
        examples, _ = read_examples_from_file(args, args.data_dir, mode)
        # print(442,examples[0].words) yes
        features = convert_examples_to_features(
            max_num_spans,# default=3 config里面help = type loss eps
            # tags_to_id,# 字典用于id与tags之间相互转换
            examples,
            args.max_seq_length,
            tokenizer,
            cls_token_at_end = False, # bool(args.TOKENIZER_TYPE in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token = tokenizer.cls_token,
            sep_token = tokenizer.sep_token,
            sep_token_extra = True, # bool(args.TOKENIZER_TYPE in ["roberta"]),2、zh这里要改
            pad_token_label_id = -100)

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)


    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # 以上nlpcc那边也有这种读取输入的代码，找到然后新增这一部分；其实只需要这两个粘过去就行
    # 在那边新增函数，然后得到all_con_spans+all_adj_matrix
    all_con_spans = torch.tensor( np.array([f.select_spans for f in features]), dtype=torch.long)# .unsqueeze(0)# tuple(4句话的spans)
    # print(469,all_con_spans.shape)# [800,502]
    all_adj_matrix = torch.tensor(np.array([f.adj_matrix for f in features]), dtype=torch.float)
    # print(471,all_adj_matrix.shape)# [800,502,502]
    # doc_id = list( [f.doc_id for f in features] )# 看下这里实现有没有问题
    doc_id = []
    for f in features:
        doc_id.append(f.doc_id)
    # print(471,doc_id)
    # 就是这里没法转为list = [str,str...]
    return doc_id,all_con_spans,all_adj_matrix


def tag_to_id(path=None, dataset=None):

    if path and os.path.exists(path+dataset+"_tag_to_id.json"):
        with open(path+dataset+"_tag_to_id.json", 'r') as f:
            data = json.load(f)
        return data # {"ner":{}, "span":{}, "type":{}}
    else:
        return None

import argparse
# preprocess_en.161 read_data调用
def get_dep_con1(tokenizer,  mode):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/nfs/home/yangliu/code/nlp/jn/secondary/project-DiaASQ/project/data/dataset/jsons_en/",
        # "data/dataset/jsons_en/",
        # "/nfs/home/yangliu/code/nlp/jn/secondary/project-DiaASQ/project/data/dataset/jsons_en/"
        type=str,
        help="",
    )
    parser.add_argument(
        "--max_seq_length",
        default=502,
        type=int,
        help="",
    )
    parser.add_argument(
        "--local_rank",
        default=-1,
        type=int,
        help="MTD的多卡模式",
    )
    parser.add_argument(
        "--seed",
        default=3407,
        type=int,
        help="",
    )
    parser.add_argument('-learning_rate_policy', '--learning_rate_policy', type=float, default=0.05, help='learning rate for iql')
    parser.add_argument('-use_softmax', '--use_softmax', type=bool, default='False', help='choose softmax')
    parser.add_argument('-policy_seed', '--policy_seed', type=int, default=0, help='random seed')
    parser.add_argument('-path', '--path', type=str, help='')
    parser.add_argument('-policy_type', '--policy_type', type=str, help='choose policy for training')
    args = parser.parse_args()

    if mode == "train":
        args.max_seq_length = 502
    elif mode== "valid":
        args.max_seq_length = 484
    else:# test
        args.max_seq_length = 481
    doc_id,spans,dep = load_and_cache_examples(args, tokenizer,  mode)
    res = {}
    res["doc_id"] = doc_id
    res["span"] = spans
    res["dep"] = dep

    return res
def get_dep_con(tokenizer,  mode,config):
    config.data_dir = "data/dataset/jsons_en/"
    # config.max_seq_length = 502
    config.local_rank = -1 # help="MTD的多卡模式",
    if mode == "train":
        config.max_seq_length = 502
    elif mode== "valid":
        config.max_seq_length = 484
    else:# test
        config.max_seq_length = 481
    # Load data features from cache or dataset file
    # args.max_tgt_num_spans = 13 # 3 # MTD default=3 config里面help = type loss eps

    doc_id,spans,dep = load_and_cache_examples(config, tokenizer,  mode)
    res = {}
    res["doc_id"] = doc_id
    res["span"] = spans
    res["dep"] = dep

    return res