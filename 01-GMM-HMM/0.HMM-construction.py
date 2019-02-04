import numpy as np

from include.fileread import read_feat, read_text
from include.env import load_env

env=load_env()
train_feat = read_feat(env["train_feat"])
train_text = read_text(env["train_text"])

"""
1. make unigram & bigram LM
"""
# Get all Monophone
monophone_set = set()
for phone_list in train_text.values():
    monophone_set.update(phone_list)

# Initialize bigram
unigram_LM = {monophone : 0 for monophone in monophone_set}
bigram_LM = {from_phone : {to_phone : 0 for to_phone in monophone_set} for from_phone in monophone_set}

# collect grammar count
for phone_list in train_text.values():
    for phone_idx in range(len(phone_list)-1):
        from_phone = phone_list[phone_idx]
        to_phone = phone_list[phone_idx+1]

        unigram_LM[from_phone] += 1
        bigram_LM[from_phone][to_phone] += 1
    unigram_LM[to_phone] += 1

# calculate bigram probability
min_prob_bound = 0.00001
for from_phone, to_phone_counts in bigram_LM.items():
    total_count = unigram_LM[from_phone]
    for to_phone, count in to_phone_counts.items():
        bigram_LM[from_phone][to_phone] = np.round(max(min_prob_bound, count / total_count) , 5)

"""
2. construct HMM state
"""
# define state class
class State:
    def __init__(self, dim=39, num_mixture=1, self_loop_prob=0.8):
        self.dim = dim
        self.num_mixture = num_mixture
        self.weight = np.full((num_mixture, 1), 1.0 / float(num_mixture))
        self.mean = np.zeros((num_mixture, dim))
        self.std = np.ones((num_mixture, dim))

        self.self_loop = self_loop_prob
        self.forward_loop = 1.0 - self_loop_prob
# construct monophone hmm
monophone_hmm_set = dict()
for monophone in monophone_set:
    monophone_hmm_set[monophone] = [ State() ] * 3

"""
3. Initialize parameter of monophone HMM
"""
# equally align monophone to feature frame
train_align = dict()
for utt_id, text_list in train_text.items():
    print(text_list)
    feat_list = train_feat[utt_id]
    total_frame_num = len(feat_list)
    align_size = int(np.floor(total_frame_num / len(text_list)))
    train_align[utt_id] = [""] * total_frame_num
    for text_idx in range(len(text_list)):
        start_idx = align_size*text_idx
        end_idx = align_size*(text_idx+1)
        text = text_list[text_idx]
        train_align[utt_id][start_idx:end_idx] = [text] * (end_idx - start_idx + 1)
    train_align[utt_id][end_idx:] = [text] * (total_frame_num - end_idx)
    assert len(train_align[utt_id]) == total_frame_num





            


