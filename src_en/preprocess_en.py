from src.utils import WordPair
import os
import re
import json

import numpy as np

from collections import defaultdict
from itertools import accumulate
from transformers import AutoTokenizer
from typing import List, Dict
from loguru import logger
from tqdm import tqdm
from src_en.data_utils import get_dep_con
import copy


# self.entity_dic = {"O": 0, "ENT-T": 1, "ENT-A": 2, "ENT-O": 3}
#
# self.rel_dic = {"O": 0, "h2h": 1, "t2t": 2}
#
# self.polarity_dic = {"O": 0, "pos": 1, "neg": 2, "other": 3}
# dididu = []  # 用于生成tokenize后的json文件

# scr_en/utils.py .40调用
class Preprocessor:
    def __init__(self, config):# yaml的config
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
        # TODO: 添加roberta-large
        # if self.config.bert_path.split("/")[-1] in ['deberta-v3-base', 'deberta-v2-xlarge', 'xlm-roberta-large', 'xlm-roberta-base','roberta-large', 'deberta-v3-base-absa-v1.1']:
        #     special_tokens_dict = {'additional_special_tokens': ['[start]','[end]']}
        #     num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        #     # model.resize_token_embeddings(len(self.tokenizer))

        self.wordpair = WordPair()
        self.entity_dict = self.wordpair.entity_dic

    def get_dict(self):
        self.polarity_dict = self.config.polarity_dict

        self.aspect_dict = {}
        for w in self.config.bio_mode:
            self.aspect_dict['{}{}'.format(w, '' if w == 'O' else '-' + self.config.asp_type)] = len(self.aspect_dict)

        self.target_dict = {}
        for w in self.config.bio_mode:
            self.target_dict['{}{}'.format(w, '' if w == 'O' else '-' + self.config.tgt_type)] = len(self.target_dict)

        self.opinion_dict = {'O': 0}
        for p in self.polarity_dict:
            if p == 'O': continue
            for w in self.config.bio_mode[1:]:
                self.opinion_dict['{}-{}_{}'.format(w, self.config.opi_type, p)] = len(self.opinion_dict)

        self.relation_dict = {'O': 0, 'yes': 1}
        return self.polarity_dict, self.target_dict, self.aspect_dict, self.opinion_dict, self.entity_dict, self.relation_dict

    def get_neighbor(self, utterance_spans, replies, max_length, speaker_ids, thread_nums):
        # utterance_mask = np.zeros([max_length, max_length], dtype=int)
        reply_mask = np.eye(max_length, dtype=int)
        for i, w in enumerate(replies):
            s1, e1 = utterance_spans[i]
            s0, e0 = utterance_spans[w + (1 if w == -1 else 0)]
            reply_mask[s0: e0 + 1, s1: e1 + 1] = 1
            reply_mask[s1: e1 + 1, s0: e0 + 1] = 1
            reply_mask[s0: e0 + 1, s0: e0 + 1] = 1
            reply_mask[s1: e1 + 1, s1: e1 + 1] = 1

        speaker_mask = np.zeros([max_length, max_length], dtype=int)
        for i, idx in enumerate(speaker_ids):
            # utterance_ids = [j for j, w in enumerate(speaker_ids) if w == idx]
            s0, e0 = utterance_spans[i]
            for j, idx1 in enumerate(speaker_ids):
                if idx != idx1: continue
                s1, e1 = utterance_spans[j]
                speaker_mask[s0: e0 + 1, s1: e1 + 1] = 1
                speaker_mask[s1: e1 + 1, s0: e0 + 1] = 1
                speaker_mask[s0: e0 + 1, s0: e0 + 1] = 1
                speaker_mask[s1: e1 + 1, s1: e1 + 1] = 1

        thread_mask = np.eye(max_length, dtype=int)
        thread_ends = accumulate(thread_nums)
        thread_spans = [(w - z, w) for w, z in zip(thread_ends, thread_nums)]
        for i, (s, e) in enumerate(thread_spans):
            if i == 0: continue
            head_start, head_end = utterance_spans[0]
            thread_mask[head_start: head_end + 1, head_start: head_end + 1] = 1
            for j in range(s, e):
                s0, e0 = utterance_spans[j]
                thread_mask[s0:e0 + 1, head_start:head_end + 1] = 1
                thread_mask[head_start:head_end + 1, s0:e0 + 1] = 1
                for k in range(s, e):
                    s1, e1 = utterance_spans[k]
                    thread_mask[s0 + 1: e0, s1 + 1: e1] = 1
                    thread_mask[s1 + 1: e1, s1 + 1: e1] = 1
                    thread_mask[s0 + 1: e0, s0 + 1: e0] = 1
                    thread_mask[s1 + 1: e1, s0 + 1: e0] = 1

        return reply_mask.tolist(), speaker_mask.tolist(), thread_mask.tolist()

    def find_utterance_index(self, replies, sentence_lengths):
        utterance_collections = [i for i, w in enumerate(replies) if w == 0]
        zero_index = utterance_collections[1]
        for i in range(len(replies)):
            if i < zero_index: continue
            if replies[i] == 0:
                zero_index = i
            replies[i] = (i - zero_index)

        sentence_index = [w + 1 for w in replies]

        utterance_index = [[w] * z for w, z in zip(sentence_index, sentence_lengths)]
        utterance_index = [w for line in utterance_index for w in line]

        token_index = [list(range(sentence_lengths[0]))]
        lens = len(token_index[0])
        for i, w in enumerate(sentence_lengths):
            if i == 0: continue
            if sentence_index[i] == 1:
                distance = lens
            token_index += [list(range(distance, distance + w))]
            distance += w
        token_index = [w for line in token_index for w in line]

        utterance_collections = np.split(sentence_index, utterance_collections)

        thread_nums = list(map(len, utterance_collections))
        thread_ranges = [0] + list(accumulate(thread_nums))
        thread_lengths = [sum(sentence_lengths[thread_ranges[i]:thread_ranges[i + 1]]) for i in
                          range(len(thread_ranges) - 1)]

        return utterance_index, token_index, thread_lengths, thread_nums

    def get_pair(self, full_triplets):
        pairs = {'ta': set(), 'ao': set(), 'to': set()}
        for i in range(len(full_triplets)):
            st, et, sa, ea, so, eo, p = full_triplets[i][:7]
            if st != -1 and sa != -1:
                pairs['ta'].add((st, et, sa, ea))

            if st != -1 and so != -1:
                pairs['to'].add((st, et, so, eo))

            if sa != -1 and eo != -1:
                pairs['ao'].add((sa, ea, so, eo))

        return pairs

    def transfer_polarity(self, pol):
        res = {'pos': 'pos', 'neg': 'neg'}
        return res.get(pol, 'other')

    def read_data(self, mode):
        """
        Read a JSON file, tokenize using BERT, and realign the indices of the original elements according to the tokenization results.
        """
        # print(150,"read {}".format(mode))
        path = os.path.join(self.config.json_path, '{}.json'.format(mode))

        if not os.path.exists(path):
            raise FileNotFoundError('File {} not found! Please check your input and data path.'.format(path))

        content = json.load(open(path, 'r', encoding='utf-8'))
        res = []  # 在这里拼入dep+con属性即可（能满足随机sample）
        # 1、这里一句句进行读取content
        json_num = 0
        # 2、在这里生成所有需要的dep+con结果数据
        dep_con = get_dep_con(self.tokenizer, mode,self.config)
        for line in tqdm(content, desc='Processing dialogues for {}'.format(mode)):
            # 3、用json_num来依次读取json数据 + 在这里依据 line["doc_id"]来看是不是读对了 ok的
            # print(dep_con['doc_id'].shape) # torch.Size([800, 502, 502])
            # print(line["doc_id"],164,dep_con['doc_id'][json_num])
            new_dialog = self.parse_dialogue(line, mode)
            # 4、设置dialogue['dep']+con的值
            new_dialog['dep'] = dep_con['dep'][json_num]  # # dep_con['dep'] = torch.Size([800, 502, 502])
            new_dialog['con'] = dep_con['span'][json_num]
            # if json_num==1 and mode=="test":
            # print(171,line["doc_id"])
            # print(172,new_dialog['con'])# torch.Size([502])
            # print(173,new_dialog['dep'][0])# torch.Size([502, 502])
            # print(174,new_dialog['dep'][1])# torch.Size([502, 502])
            # print(175,new_dialog['dep'][2])# torch.Size([502, 502])
            json_num += 1  # 读下一句的dep+con
            res.append(new_dialog)
        # 保存tokenize结果dididu到json文件中：
        # with open(
        #         "/nfs/home/yangliu/code/nlp/jn/secondary/project-DiaASQ/project/data/dataset/yjj/En_{}_yjj.json".format(
        #                 mode), 'w', encoding='utf-8') as f:
        #     json.dump(dididu, f, indent=4)
        # dididu.clear()
        # print("{} sub-word over".format(mode))

        # 这里加入读取得到con_head+con_dep的代码

        return res

    def check_text(self, tokenized_text, source_text):
        roberta_list = ['roberta-large', 'roberta-base', 'roberta-large-ner-english']
        # roberta_list = ['xlm-roberta-large', 'roberta-large', 'roberta-base', 'roberta-large-ner-english']
        if self.config.bert_path.split('/')[-1] in roberta_list:
            t0 = tokenized_text.lower()
            roberta_chars = 'â ī ¥ Ġ ð ł ĺ ħ Ł ŀ į Ŀ Į ĵ © ĵ ĳ ¶ ã'.split()
            unused = [self.config.unk, '##']
            if self.config.bert_path.split('/')[-1] in roberta_list:
                unused += roberta_chars
            for u in unused:
                t0 = t0.replace(u.lower(), '')
            t1 = source_text.replace(' ', '').lower()
            for k in self.config.unkown_tokens:
                t1 = t1.replace(k, '')
            if self.config.bert_path.split('/')[-1] in roberta_list:
                t1 = t1.replace('×', '').replace('≥', '')
            if t0 != t1:
                logger.info(t1 + '||' + t1)
                logger.info(tokenized_text + '||' + source_text)
                t2 = t0
                for u in unused:
                    t2 = t2.replace(u, '')
                raise AssertionError("--{}-- != --{}--".format(t0, t1))
            return t0 == t1
        spm_list = ['deberta-v3-base', 'deberta-v2-xlarge', 'xlm-roberta-large', 'xlm-roberta-base',
                    'deberta-v3-base-absa-v1.1']
        # spm_list = ['deberta-v3-base', 'deberta-v2-xlarge', 'deberta-v3-base-absa-v1.1']
        if self.config.bert_path.split('/')[-1] in spm_list:
            t0 = tokenized_text.replace('▁', '').replace(self.config.unk, '').lower()
            t1 = source_text.replace(' ', '').lower()
            for k in self.config.unkown_tokens:
                t1 = t1.replace(k, '')
            if t0 != t1:
                logger.info(t1 + '||' + t1)
                logger.info(tokenized_text + '||' + source_text)
                raise AssertionError("{} != {}".format(t0, t1))
            return t0 == t1

        t0 = tokenized_text.replace('##', '').replace(self.config.unk, '').lower()
        t1 = source_text.replace(' ', '').lower()
        for k in self.config.unkown_tokens:
            t1 = t1.replace(k, '')
        if t0 != t1:
            logger.info(t1 + '||' + t1)
            logger.info(tokenized_text + '||' + source_text)
            raise AssertionError("{} != {}".format(t0, t1))
        return t0 == t1

    def parse_dialogue(self, dialogue, mode):
        # get the list of sentences in the dialogue
        sentences = dialogue['sentences']
        doc_id = dialogue['doc_id']
        # align_index_with_list: align the index of the original elements according to the tokenization results
        new_sentences, pieces2words = self.align_index_with_list(doc_id, sentences)

        word2pieces = defaultdict(list)
        for p, w in enumerate(pieces2words):
            word2pieces[w].append(p)

        dialogue['pieces2words'] = pieces2words
        dialogue['sentences'] = new_sentences

        # get target, aspect and opinion respectively, and align to the new index

        if mode != 'train':
            return dialogue
        targets, aspects, opinions = [dialogue[w] for w in ['targets', 'aspects', 'opinions']]
        targets = [(word2pieces[x][0], word2pieces[y - 1][-1] + 1, z) for x, y, z in targets]
        aspects = [(word2pieces[x][0], word2pieces[y - 1][-1] + 1, z) for x, y, z in aspects]
        opinions = [(word2pieces[x][0], word2pieces[y - 1][-1] + 1, z, self.transfer_polarity(w)) for x, y, z, w in
                    opinions]

        # Put the elements into the dialogue object after converting the elements to the new index
        dialogue['targets'], dialogue['aspects'], dialogue['opinions'] = targets, aspects, opinions

        # Flatten the two-dimensional list and put the entire dialogue in a list
        news = [w for line in new_sentences for w in line]

        # Confirm the index again
        for ts, te, t_t in targets:
            assert self.check_text(''.join(news[ts:te]), t_t)
        for ts, te, t_t in aspects:
            assert self.check_text(''.join(news[ts:te]), t_t)
        for ts, te, t_t, _ in opinions:
            assert self.check_text(''.join(news[ts:te]), t_t)

        triplets = []

        # polarity transfer and index transfer

        for t_s, t_e, a_s, a_e, o_s, o_e, polarity, t_t, a_t, o_t in dialogue['triplets']:
            polarity = self.transfer_polarity(polarity)
            nts, nas, nos = [word2pieces[w][0] if w != -1 else -1 for w in [t_s, a_s, o_s]]
            nte, nae, noe = [word2pieces[w - 1][-1] + 1 if w != -1 else -1 for w in [t_e, a_e, o_e]]
            self.check_text(''.join(news[nts:nte]), t_t)
            self.check_text(''.join(news[nas:nae]), a_t) or nas == -1
            if not self.check_text(''.join(news[nos:noe]), o_t) and nos != -1:
                logger.info(''.join(news[nos:noe]) + '||' + o_t)
            self.check_text(''.join(news[nos:noe]), o_t) or nos == -1

            triplets.append((nts, nte, nas, nae, nos, noe, polarity, t_t, a_t, o_t))
        dialogue['triplets'] = triplets
        return dialogue

        # 实现分词的函数

    def align_index_with_list(self, doc_id, sentences):
        """_summary_
        align the index of the original elements according to the tokenization results
        Args:
            sentences (_type_): List<str>
            e.g., xiao mi 12x is my favorite
        """
        # cur_line = []  # 改这里直接得到想要的[各个分词数据]
        # all_pieces = []
        # for sentence in sentences:
        #     sentence = sentence.split()
        #     tokens = [self.tokenizer.tokenize(w) for w in sentence]
        #     for token in tokens:
        #         cur_line += token
        # all_pieces.append(cur_line)
        # didi = {"doc_id": doc_id, "str_words": all_pieces[0]}
        # dididu.append(didi)
        # print(308,dididu)

        # 以下是原代码
        pieces2word = []
        word_num = 0
        all_pieces = []
        # 改这里直接得到想要的[各个分词数据] 原代码：
        for sentence in sentences:  # 依次遍历每句话
            sentence = sentence.split()
            tokens = [self.tokenizer.tokenize(w) for w in sentence]  # 分词
            cur_line = []
            for token in tokens:
                for piece in token:
                    pieces2word.append(word_num)
                word_num += 1
                cur_line += token
            all_pieces.append(cur_line)

        return all_pieces, pieces2word  # 所以是这里需要还原为word级别的话/直接不使用tokenizer分词
        # all_pieces 作为 new—sentences   后面那个是每句话token数量

    def align_index(self, sentences):
        res, char2token = [], {}
        source_lens, token_lens = 0, 0
        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)
            if self.config.bert_path in ['roberta-large', 'bert-base-uncased']:
                c2t, tokens = self.alignment_roberta(sentence, tokens)
            else:
                c2t, tokens = self.alignment(sentence, tokens)
            res.append(tokens)
            for k, v in c2t.items():
                char2token[k + source_lens] = v + token_lens
            source_lens, token_lens = source_lens + len(sentence) + 1, token_lens + len(tokens)

        return res, char2token

    def alignment(self, source_sequence, tokenized_sequence: List[str], align_type: str = 'one2many') -> Dict:
        """[summary]
        # this is a function that to align sequcences  that before tokenized and after.
        Parameters
        ----------
        source_sequence : [type]
            this is the original sequence, whose type either can be str or list
        tokenized_sequence : List[str]
            this is the tokenized sequcen, which is a list of tokens.
        index_type : str, optional, default: str
            this indicate whether source_sequence is str or list, by default 'str'
        align_type : str, optional, default: one2many
            there may be several kinds of tokenizer style,
            one2many: one word in source sequence can be split into multiple tokens
            many2one: many word in source sequence will be merged into one token
            many2many: both contains one2many and many2one in a sequence, this is the most complicated situation.

        useage:
        source_sequence = "Here, we investigate the structure and dissociation process of interfacial water"
        tokenized_sequence = ['here', ',', 'we', 'investigate', 'the', 'structure', 'and', 'di', '##sso', '##ciation', 'process', 'of', 'inter', '##fa', '##cial', 'water']
        char2token = alignment(source_sequence, tokenized_sequence)
        print(char2token)
        for c, t in char2token.items():
            print(source_sequence[c], tokenized_sequence[t])
        """
        char2token = {}
        if isinstance(source_sequence, str) and align_type == 'one2many':
            source_sequence = source_sequence.lower()
            i, j = 0, 0
            while i < len(source_sequence) and j < len(tokenized_sequence):
                cur_token, length = tokenized_sequence[j], len(tokenized_sequence[j])
                if source_sequence[i] == ' ':
                    i += 1
                elif source_sequence[i: i + length] == cur_token:
                    for k in range(length):
                        char2token[i + k] = j
                    i, j = i + length, j + 1
                elif tokenized_sequence[j] == self.config.unk:
                    lens = 1
                    if j + 1 == len(tokenized_sequence):
                        lens = len(source_sequence) - i
                    else:
                        while i + lens < len(source_sequence):
                            if source_sequence[i + lens] == tokenized_sequence[j + 1].strip('#')[0] or \
                                    tokenized_sequence[j + 1] == self.config.unk:
                                break
                            lens += 1
                    new_token = self.repack_unknow(source_sequence[i:i + lens])
                    tokenized_sequence = tokenized_sequence[:j] + new_token + tokenized_sequence[j + 1:]
                    if tokenized_sequence[j] == self.config.unk:
                        char2token[i] = j
                        i += 1
                        j += 1
                else:
                    assert tokenized_sequence[j].startswith('#')
                    length = len(tokenized_sequence[j].lstrip('#'))
                    assert source_sequence[i: i + length] == tokenized_sequence[j].lstrip('#')
                    for k in range(length):
                        char2token[i + k] = j
                    i, j = i + length, j + 1
        return char2token, tokenized_sequence

    def alignment_roberta(self, source_sequence, tokenized_sequence: List[str]) -> Dict:
        # For English dataset
        char2token = {}
        if isinstance(source_sequence, str):
            source_sequence = source_sequence.lower()
            i, j = 0, 0
            while i < len(source_sequence) and j < len(tokenized_sequence):
                cur_token, length = tokenized_sequence[j], len(tokenized_sequence[j].strip('Ġ'))
                if source_sequence[i] == ' ':
                    i += 1
                elif source_sequence[i: i + length].lower() == cur_token.strip('Ġ').lower():
                    for k in range(length):
                        char2token[i + k] = j
                    i, j = i + length, j + 1
                elif tokenized_sequence[j] == self.config.unk:
                    lens = 1
                    if j + 1 == len(tokenized_sequence):
                        lens = len(source_sequence) - i
                    else:
                        while i + lens < len(source_sequence):
                            if source_sequence[i + lens] == tokenized_sequence[j + 1].strip('#')[0] or \
                                    tokenized_sequence[j + 1] == self.config.unk:
                                if tokenized_sequence[j + 1].strip('#')[0] == 'i' and j + 1 < len(
                                        tokenized_sequence) and len(tokenized_sequence[j + 1].strip()) > 1:
                                    if i + lens + 1 < len(source_sequence) and source_sequence[i + lens + 1] == \
                                            tokenized_sequence[j + 1].strip('#')[1]:
                                        break
                                else:
                                    break
                            lens += 1
                    new_token = self.repack_unknow(source_sequence[i:i + lens])
                    tokenized_sequence = tokenized_sequence[:j] + new_token + tokenized_sequence[j + 1:]
                    if tokenized_sequence[j] == self.config.unk:
                        char2token[i] = j
                        i += 1
                        j += 1
                else:
                    assert tokenized_sequence[j].startswith('#')
                    length = len(tokenized_sequence[j].lstrip('#'))
                    assert source_sequence[i: i + length] == tokenized_sequence[j].lstrip('#')
                    for k in range(length):
                        char2token[i + k] = j
                    i, j = i + length, j + 1
        return char2token, tokenized_sequence

    def repack_unknow(self, source_sequence):
        '''
        # sentence='🍎12💩', Bert can't recognize two contiguous emojis, so it recognizes the whole as '[UNK]'
        # We need to manually split it, recognize the words that are not in the bert vocabulary as UNK,
        and let BERT re-segment the parts that can be recognized, such as numbers
        # The above example processing result is: ['[UNK]', '12', '[UNK]']
        '''
        lst = list(re.finditer('|'.join(self.config.unkown_tokens), source_sequence))
        start, i = 0, 0
        new_tokens = []
        while i < len(lst):
            s, e = lst[i].span()
            if start < s:
                token = self.tokenizer.tokenize(source_sequence[start:s])
                new_tokens += token
                start = s
            else:
                new_tokens.append(self.config.unk)
                start = e
            i += 1
        if start < len(source_sequence):
            token = self.tokenizer.tokenize(source_sequence[start:])
            new_tokens += token
        return new_tokens

    def transform2indices(self, dataset, mode='train'):
        res = []
        max_len_recoder = 0
        # 在for循环里面得到dep+con，然后直接在这里面加入对应的dep+con即可
        # 或者read_data里面得到，然后在这里根据doc_id读取数据的dep+con也可！！！
        for document in dataset:  # 这里面句子也都是按json文件顺序的
            # 这里的document就是read_data函数里的new_dialog
            sentences, speakers, replies, pieces2words = [document[w] for w in
                                                          ['sentences', 'speakers', 'replies', 'pieces2words']]
            if mode == 'train':
                # 读取最原始的部分
                triplets, targets, aspects, opinions = [document[w] for w in
                                                        ['triplets', 'targets', 'aspects', 'opinions']]
            doc_id = document['doc_id']
            # if mode == "train":
            #     if doc_id=='0428':
            # print(428,doc_id,428,sentences) # ,428,input_ids
            # orginal_replies = copy.deepcopy(replies)
            # sentence_length = list(map(lambda x : len(x) + 2, sentences))
            sentence_length = list(map(lambda x: len(x) + 2, sentences))

            # token2sentid = [[i] * len(w) for i, w in enumerate(sentences)]
            token2sentid = [[i] * len(w) for i, w in enumerate(sentences)]
            token2sentid = [w for line in token2sentid for w in line]

            token2speaker = [[11] + [w] * len(z) + [10] for w, z in zip(speakers, sentences)]
            token2speaker = [w for line in token2speaker for w in line]

            # New token indices (with CLS and SEP) to old token indices (without CLS and SEP)
            new2old = {}
            cur_len = 0
            for i in range(len(sentence_length)):
                for j in range(sentence_length[i]):
                    if j == 0 or j == sentence_length[i] - 1:
                        new2old[len(new2old)] = -1
                    else:
                        new2old[len(new2old)] = cur_len
                        cur_len += 1
            # tokens = []
            # for index in range(len(sentences)):
            #     if index == 0:
            #         tokens.append([self.config.cls] + w + [self.config.end] for w in sentences)
            #     elif index == len(sentences)-1:
            #         tokens.append([self.config.start] + w + [self.config.sep] for w in sentences)
            #     else:
            #         tokens.append([self.config.start] + w + [self.config.end] for w in sentences)
            tokens = [[self.config.cls] + w + [self.config.sep] for w in sentences]
            multi_tokens = [[self.config.cls] + w + [self.config.sep] for w in sentences]

            # sentence_ids of each token (new token)
            nsentence_ids = [[i] * len(w) for i, w in enumerate(tokens)]
            nsentence_ids = [w for line in nsentence_ids for w in line]

            flatten_tokens = [w for line in tokens for w in line]
            sentence_end = [i - 1 for i, w in enumerate(flatten_tokens) if w == self.config.sep]
            sentence_start = [i + 1 for i, w in enumerate(flatten_tokens) if w == self.config.cls]

            utterance_spans = list(zip(sentence_start, sentence_end))
            utterance_index, token_index, thread_length, thread_nums = self.find_utterance_index(replies,
                                                                                                 sentence_length)
            reply_mask, speaker_masks, thread_masks = self.get_neighbor(utterance_spans, replies, sum(sentence_length),
                                                                        speakers, thread_nums)
            # reply_mask, speaker_masks, thread_masks = self.get_neighbor(utterance_spans, orginal_replies, sum(sentence_length), speakers, thread_nums)

            tokens = [[self.config.start] + w + [self.config.end] for w in sentences]
            tokens[0][0], tokens[-1][-1] = self.config.cls, self.config.sep

            input_ids = list(map(self.tokenizer.convert_tokens_to_ids, tokens))
            input_masks = [[1] * len(w) for w in input_ids]  # 先全置为1
            input_segments = [[0] * len(w) for w in input_ids]
            input_ids = [[w for line in input_ids for w in line]]
            input_masks = [[w for line in input_masks for w in line]]
            input_segments = [[w for line in input_segments for w in line]]

            multi_input_ids = list(map(self.tokenizer.convert_tokens_to_ids, multi_tokens))
            multi_input_masks = [[1] * len(w) for w in multi_input_ids]
            multi_input_segments = [[0] * len(w) for w in multi_input_ids]

            if len(input_ids[0]) > max_len_recoder:
                max_len_recoder = len(input_ids[0])
            if mode == 'train':  # s e即start end 的值是有一定程度偏移的
                targets = [(s + 2 * token2sentid[s] + 1, e + 2 * token2sentid[s]) for s, e, t in targets]
                aspects = [(s + 2 * token2sentid[s] + 1, e + 2 * token2sentid[s]) for s, e, t in aspects]
                opinions = [(s + 2 * token2sentid[s] + 1, e + 2 * token2sentid[s]) for s, e, t, p in opinions]
                opinions = list(set(opinions))

                full_triplets, new_triplets = [], []
                # t_s-> target_start, t_e-> target_end, etc.
                for t_s, t_e, a_s, a_e, o_s, o_e, polarity, t_t, a_t, o_t in triplets:
                    # 107,108,109,112,106,107,      "pos",          "Meizu",  "small window mode",  "like"
                    # polarity      target       aspect            opinion

                    new_index = lambda start, end: (-1, -1) if start == -1 else (
                    start + 2 * token2sentid[start] + 1, end + 2 * token2sentid[start])
                    t_s, t_e = new_index(t_s, t_e)  # 得到新的索引
                    a_s, a_e = new_index(a_s, a_e)
                    o_s, o_e = new_index(o_s, o_e)
                    # polarity_dict:    O: 0    pos: 1  neg: 2  other: 3
                    line = (t_s, t_e, a_s, a_e, o_s, o_e, self.polarity_dict[polarity])
                    full_triplets.append(line)
                    if all(w != -1 for w in [t_s, a_s, o_s]):
                        new_triplets.append(line)

                relation_lists = self.wordpair.encode_relation(full_triplets)
                pairs = self.get_pair(full_triplets)

                target_lists = self.wordpair.encode_entity(targets,
                                                           'ENT-T')  # self.entity_dic = {"O": 0, "ENT-T": 1, "ENT-A": 2, "ENT-O": 3}
                aspect_lists = self.wordpair.encode_entity(aspects, 'ENT-A')  #
                opinion_lists = self.wordpair.encode_entity(opinions, 'ENT-O')

                entity_lists = target_lists + aspect_lists + opinion_lists
                polarity_lists = self.wordpair.encode_polarity(new_triplets)
                # 函数 return (starting position, ending position, polarity category）
            else:
                new_triplets, pairs, entity_lists, relation_lists, polarity_lists = [], [], [], [], []
            # res.append((doc_id, input_ids, input_masks, input_segments, sentence_length, nsentence_ids, utterance_index, token_index,
            #             thread_length, token2speaker, reply_mask, speaker_masks, thread_masks, pieces2words, new2old,
            #             new_triplets, pairs, entity_lists, relation_lists, polarity_lists))

            # res.append((doc_id, input_ids, input_masks, input_segments, sentence_length, nsentence_ids, utterance_index, token_index,
            #             thread_length, token2speaker, reply_mask, speaker_masks, thread_masks, pieces2words, new2old,
            #             new_triplets, pairs, entity_lists, relation_lists, polarity_lists, multi_input_ids, multi_input_masks, multi_input_segments))
            if len(input_ids[0]) <= 512:
                # 5、在这里设置res['dep']的值即可
                # doc_id = document['doc_id']
                con = document['con']
                dep = document['dep']
                res.append((doc_id, input_ids, input_masks, input_segments, sentence_length, nsentence_ids,
                            utterance_index, token_index,
                            thread_length, token2speaker, reply_mask, speaker_masks, thread_masks, pieces2words,
                            new2old,
                            new_triplets, pairs, entity_lists, relation_lists, polarity_lists, multi_input_ids,
                            multi_input_masks, multi_input_segments,
                            con, dep
                            ))
            # print(565,doc_id)# ,entity_lists
            # 565 0525 Target: [(82, 82, 1), (138, 139, 1), (157, 158, 1), (170, 171, 1), (201, 202, 1),
            # Aspect:  (141, 143, 2), (146, 146, 2), (86, 86, 2), (88, 90, 2), (215, 216, 2), (224, 226, 2),
            # Opinion: (228, 231, 3), (42, 43, 3), (172, 174, 3), (92, 94, 3), (137, 137, 3), (207, 208, 3), (147, 150, 3), (219, 220, 3), (160, 162, 3)]

            # 566 0525    O: 0  pos: 1  neg: 2  other: 3
            # [(138, 137, 1), (139, 137, 1), (138, 147, 1), (139, 150, 1), (157, 160, 1), (158, 162, 1),
            # (201, 219, 3), (202, 220, 3),
            # (201, 228, 2), (202, 231, 2), (82, 92, 2), (82, 94, 2)]
            # print(566,doc_id,polarity_lists)

            # if mode == "train":
            #     if doc_id=='0428':
            #         print(428,doc_id,428,sentences) # ,428,input_ids
            #     elif  doc_id=='0798':
            #         print(798, doc_id, 798, sentences, 798, input_ids)
            #     elif doc_id == '0426':
            #         print(426, doc_id, 426, sentences, 426, input_ids)
            #     elif doc_id == '0987':
            #         print(987, doc_id, 987, sentences, 987, input_ids)
        print("{} max len input_ids is {}".format(mode, max_len_recoder))

        return res

    def forward(self):
        # modes default: 'train valid test'
        modes = self.config.input_files
        # modes = "test"
        datasets = {}
        # print(629)
        for mode in modes.split():
            data = self.read_data(mode)  # 这里应该是最原始的数据
            datasets[mode] = data
        # print(633)
        label_dict = self.get_dict()  # 字典

        res = {}
        for mode in modes.split():  # 还需要这里读入dep+con；以及最后实现随机sample的那个函数看看也需不需要改一下
            res[mode] = self.transform2indices(datasets[mode], mode)
        res['label_dict'] = label_dict
        return res