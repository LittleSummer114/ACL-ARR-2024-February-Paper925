
policy_type='iql'

# 前8张调seed ? 

# learning_rate_policy 跑了1.00e-06, 2.00e-06, 1.00e-05, 2.00e-05,设置为2.00e-05
# af0,af1,af2跑了(0.3,0.4,0.3)(0.6,0.2,0.2),(0.4,0.3,0.3),(0.3,0.4,0.3),(0.2,0.6,0.2),(0.3,0.3,0.4),(0.2,0.2,0.6),设置为0.3,0.4,0.3

CUDA_VISIBLE_DEVICES=0 python /nfs/home/yangliu/code/nlp/jn/secondary/project-DiaASQ/project/main_en_train.py \

--seed 0 \ 之前42、3407这两个seed比较好
--policy_seed 14 \  这个也是个随机seed
--bert_lr 1e-5 \ 两个学习率也能调 如1e-3、1e-4 、1e-5 、 1e-6
--learning_rate_policy 1e-5 \ 两个学习率也能调 如1e-3、1e-4 、1e-5 、 1e-6
--policy_type $policy_type \ 这个选强化学习算法的 不能动
--use_softmax 0 \ 可取[0/1] 之前实验了是0较好
--max_tgt_num_spans 13 \ 这个不要动，我记得en是13，zh是10最好
--epoch_size 50 \ epoch
--reward_weights 1 \ 可取的值很多，如-1、1、10、18、100....反正各种值都能取
--af0 0.3 \ af0/1/2  这三个取值之和要是1  之前实验的三个差不多值效果最好
--af1 0.4 \
--af2 0.3 \

CUDA_VISIBLE_DEVICES=1 python /nfs/home/yangliu/code/nlp/jn/secondary/project-DiaASQ/project/main_en_train.py \

--seed 1 \ 之前42、3407这两个seed比较好
--policy_seed 14 \  这个也是个随机seed
--bert_lr 1e-5 \ 两个学习率也能调 如1e-3、1e-4 、1e-5 、 1e-6
--learning_rate_policy 1e-5 \ 两个学习率也能调 如1e-3、1e-4 、1e-5 、 1e-6
--policy_type $policy_type \ 这个选强化学习算法的 不能动
--use_softmax 0 \ 可取[0/1] 之前实验了是0较好
--max_tgt_num_spans 13 \ 这个不要动，我记得en是13，zh是10最好
--epoch_size 50 \ epoch
--reward_weights 1 \ 可取的值很多，如-1、1、10、18、100....反正各种值都能取
--af0 0.3 \ af0/1/2  这三个取值之和要是1  之前实验的三个差不多值效果最好
--af1 0.4 \
--af2 0.3 \

CUDA_VISIBLE_DEVICES=2 python /nfs/home/yangliu/code/nlp/jn/secondary/project-DiaASQ/project/main_en_train.py \

--seed 2 \ 之前42、3407这两个seed比较好
--policy_seed 14 \  这个也是个随机seed
--bert_lr 1e-5 \ 两个学习率也能调 如1e-3、1e-4 、1e-5 、 1e-6
--learning_rate_policy 1e-5 \ 两个学习率也能调 如1e-3、1e-4 、1e-5 、 1e-6
--policy_type $policy_type \ 这个选强化学习算法的 不能动
--use_softmax 0 \ 可取[0/1] 之前实验了是0较好
--max_tgt_num_spans 13 \ 这个不要动，我记得en是13，zh是10最好
--epoch_size 50 \ epoch
--reward_weights 1 \ 可取的值很多，如-1、1、10、18、100....反正各种值都能取
--af0 0.3 \ af0/1/2  这三个取值之和要是1  之前实验的三个差不多值效果最好
--af1 0.4 \
--af2 0.3 \