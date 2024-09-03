import numpy as np
import json
import os

def read_data(path, pd=None):
    with open(path, 'r', encoding='utf-8') as f:
        content = json.load(f)
    res = {}
    if pd is not None:
        content = [w for w in content if w['doc_id'] in pd]
    for line in content:
        doc_id = line['doc_id']
        triplets = line['triplets']
        cur_res = set()
        for comb in triplets:
            if any(w == -1 for w in comb[:6]): continue
            assert all(isinstance(w, int) for w in comb[:6])
            assert isinstance(comb[6], str)
            assert len(comb) == 10
            comb[6] = comb[6] if comb[6] in ['pos', 'neg'] else 'other'
            cur_res.add(tuple(comb[:7]))
        cur_res = sorted(list(cur_res), key=lambda x: (x[0], x[2], x[4]))
        res[doc_id] = cur_res

    return res

# 这个函数能计算F1的前提是结果为4元组，否则要改改函数
def post_process(line, key='quad'): # sentiments,targets, aspects, opinions
    # if key in ['targets', 'aspects', 'opinions']:
    #     return [tuple(w[:2]) for w in line[key]]
    res = []
    if key in ['targets', 'aspects', 'opinions']:
        if key == 'targets':
            # print("targets_score:")
            # for comb in line['triplets']:
                # print(comb[1])
                # comb[0] = comb[0] if comb[0] in ['pos', 'neg'] else 'other'
                # res.append(tuple( (comb[1:2])))
            res = [w[:2] for w in line]
            return res

        elif key == 'aspects':
            return [w[2:4] for w in line]

        elif key == 'opinions':
            return [w[4:6] for w in line]
            # return [tuple(line[3])]

#   0 1 2 3 4 5 pos
    if key in ['ta', 'to', 'ao']:# 0 targets, aspects, opinions
        if key == 'ta': # target + aspect
            # for comb in line['triplets']:
                # print(comb[1])
                # res.append(tuple( (comb[1:3])))
            # return res
            return [w[0:4] for w in line]#   0 1 2 3 4 5 pos
            # return [tuple(line[1:3])]# 1+2
        elif key == 'to':# target + opinion
            # for comb in line['triplets']:
                # print(comb[1])
                # res.append(tuple( (comb[1:2] + comb[3:4])))
            res = [w[0:2] + w[4:6] for w in line]
            return res
            # return [tuple( (line[1] + line[3] ) )]# 1+3
        elif key == 'ao':# # aspect + opinion
            # for comb in line['triplets']:
                # print(comb[1])
                # res.append(tuple( (comb[2:4])))
            return [w[2:6] for w in line]
            # return [tuple(line[2:4])]# 2+3
        # return [tuple(w[:4]) for w in line[key]]

    if key in ['quad', 'Iden']:
        if key == 'quad':
            return line
            # for comb in line['triplets']:
            #     # if any(w == -1 for w in comb[:6]): continue
            #     # assert all(isinstance(w, int) for w in comb[:6])
            #     # assert isinstance(comb[6], str)
            #     # assert len(comb) == 10
            #
            #     # 注意这里，处理amb和doubt为other了(gold/pred)
            #     comb[0] = comb[0] if comb[0] in ['pos', 'neg'] else 'other'
            #     res.append(tuple(comb))
            #
            # return res
        elif key == 'Iden': # t + a + o
            res = [w[:6] for w in line]
            # for comb in line['triplets']:
                # print(comb[1])
                # res.append(tuple((comb[1:4])))
            return res
    # if key in ['intra', 'inter']:
    #     for comb in line['triplets']:
    #         if any(w == -1 for w in comb[:6]): continue
    #         comb[6] = comb[6] if comb[6] in ['pos', 'neg'] else 'other'
    #         distance = get_utterance_distance(line['sentence_ids'], line['dis_matrix'], comb[0], comb[2], comb[4])
    #         if key == 'intra' and distance > 0: continue
    #         if key == 'inter' and distance == 0: continue
    #         res.append(tuple(comb[:7]))
    #     return res
    raise ValueError('Invalid key: {}'.format(key))

def compute_score(gold_res,pred_res,mode='quad'):  # ['targets', 'aspects', 'opinions', 'ta', 'to', 'ao', 'quad', 'iden', 'intra', 'inter']
    # print(117,mode) # mode用于区分各个数据集    quad = triplets
    tp, fp, fn = 0, 0, 0
    # for doc_id in range(100):
    for doc_id in pred_res:
        # print(f"第{doc_id}句话：")
        # if (mode == 'quad'):
        #     print(doc_id, 90, gold_res[doc_id]['triplets'])
        # [[0, 2, 86, 87, 88, 91, 'neg'],[]]
        pred_line = pred_res[doc_id]
        # if mode in ['ta','to','ao']:
        #     print(116,pred_line)
        gold_line = gold_res[doc_id]

        # pred_line['sentence_ids'] = gold_line['sentence_ids']
        # pred_line['dis_matrix'] = gold_line['dis_matrix']

        pred_line = post_process(pred_line, mode)  # 处理了读取 -> [(0, 2, 86, 87, 88, 91, 'neg')]
        # if mode in ['ta','to','ao']:
        #     print("123 {}:".format(mode),pred_line)
        gold_line = post_process(gold_line, mode)
        # print("gold:",gold_line)
        # print(gold_line) #[(0, 1, 155, 156, 156, 157, 'pos'), (0, 1, 143, 145, 145, 148, 'neg')]
        fp += len(set(pred_line) - set(gold_line))
        # print(54,fp)
        fn += len(set(gold_line) - set(pred_line))
        # print(55, fn)
        # if (mode == 'quad'):
        #     print(doc_id, 123, set(gold_line))
        # print(gold_line) # [(0, 2, 86, 87, 88, 91, 'neg')]
        # print(set(gold_line)) # {(0, 2, 86, 87, 88, 91, 'neg')}
        tp += len(set(pred_line) & set(gold_line))
        # print(62,tp)

    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    scores = [p, r, f1]
    return scores


def record_F(config,lang):# 传入config.target_dir+lang
    mode = "test"
    pred_path = config.target_dir+'/pred_{}_{}.json'.format(lang, mode)
    if not os.path.exists(pred_path):
    # mode = "valid"
        print("not find",pred_path)
    # a1 = 'D:\桌面/10.20 ACL\论文\第二名\project-DiaASQ\project\data/run-eval/{}'.format(lang)
    # a1 = "D:\桌面"
    # a1 += '/pred_{}_{}.json'.format(lang, mode)
    content_pred = read_data(pred_path)

    gold_path = '/nfs/home/yangliu/code/nlp/jn/secondary/project-DiaASQ/project/record/{}/test.json'.format(lang)
    content_gold = read_data(gold_path, pd=content_pred)

    f = open(config.target_dir+'/pred_score.txt', 'w')# +"-" + str(config.seed)+"-"
    f.write("p \t r \t F1 \n")
    for key in  ['quad','Iden']:
        score = compute_score(content_gold,content_pred,mode=key)
        f.write( '{}: \t {}% \t {}% \t {}% \n'.format(key,score[0]*100,score[1]*100,score[2]*100) )
    f.close()


    save_name = str(config.seed)+"-"+str(config.policy_seed)
    save_name += "-"+str(config.bert_lr)+"-"+str(config.learning_rate_policy)+"-"+str(config.policy_type)+"-"+str(config.use_softmax)
    save_name+= "-"+str(config.max_tgt_num_spans)+"-"+str(config.epoch_size)+"-"+str(config.reward_weights)
    save_name += "-" + str(config.af0) + "-" + str(config.af1) + "-" + str(config.af2)
    f = open("/nfs/home/yangliu/code/nlp/jn/secondary/project-DiaASQ/project/record/{}/results".format(config.lang)+'/{}.txt'.format(save_name), 'w')
    f.write ("seed {} \t policy_seed {} \n".format(config.seed,config.policy_seed) )
    f.write ("bert_lr {} \t learning_rate_policy {} \n".format(config.bert_lr,config.learning_rate_policy) )
    f.write ("policy_type {} \t use_softmax {} \n".format(config.policy_type,config.use_softmax) )
    f.write ("max_tgt_num_spans {} \t epoch_size {} \n".format(config.max_tgt_num_spans,config.epoch_size) )
    f.write ("reward_weights {}     \n".format(config.reward_weights) )
    f.write("p \t r \t F1 \n")

    for key in  ['quad','Iden']:
        score = compute_score(content_gold,content_pred,mode=key)
        f.write( '{}: \t {}% \t {}% \t {}% \n'.format(key,score[0]*100,score[1]*100,score[2]*100) )
    f.close()

    for key in  ['quad','Iden']:
        score = compute_score(content_gold,content_pred,mode=key)
        if(key=='quad'):
            if lang=="en":
                if (score[2] * 100 > 41.84):
                    f = open('/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/coco/ACL/project/record/{}'.format(lang)+ '/best_score.txt', 'a')
                    f.write(config.seed,config.policy_seed,score[2]*100)
                    f.write('\n')
                    f.close()
            if lang=="zh":
                if (score[2] * 100 > 48.26):
                    f = open(
                        '/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/coco/ACL/project/record/{}'.format(
                            lang) + '/best_score.txt', 'a')
                    f.write(config.seed, config.policy_seed,score[2]*100)
                    f.write('\n')
                    f.close()