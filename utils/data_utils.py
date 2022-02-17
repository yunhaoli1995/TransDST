"""
SOM-DST
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

import numpy as np
import json
from torch.utils.data import Dataset
import torch
import random
import math
import re
from copy import deepcopy
from .fix_label import fix_general_label_error
import os
import pickle
import logging
from tqdm import tqdm

flatten = lambda x: [i for s in x for i in s]
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
domain2id = {d: i for i, d in enumerate(EXPERIMENT_DOMAINS)}

OP_SET = {
    '2': {'update': 0, 'carryover': 1},
    '3-1': {'update': 0, 'carryover': 1, 'dontcare': 2},
    '3-2': {'update': 0, 'carryover': 1, 'delete': 2},
    '4': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3},
    '6': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3, 'yes': 4, 'no': 5}
}

def make_slot_meta(ontology):
    meta = []
    change = {}
    idx = 0
    max_len = 0
    for i, k in enumerate(ontology.keys()):
        d, s = k.split('-')
        if d not in EXPERIMENT_DOMAINS:
            continue
        if 'price' in s or 'leave' in s or 'arrive' in s:
            s = s.replace(' ', '')
        ss = s.split()
        if len(ss) + 1 > max_len:
            max_len = len(ss) + 1
        meta.append('-'.join([d, s]))
        change[meta[-1]] = ontology[k]
    return sorted(meta), change
    
def make_turn_label(slot_meta, last_dialog_state, turn_dialog_state,
                    tokenizer, op_code='4', dynamic=False):
    if dynamic:
        gold_state = turn_dialog_state
        turn_dialog_state = {}
        for x in gold_state:
            s = x.split('-')
            k = '-'.join(s[:2])
            turn_dialog_state[k] = s[2]

    op_labels = ['carryover'] * len(slot_meta)
    generate_y = []
    keys = list(turn_dialog_state.keys())
    for k in keys:
        v = turn_dialog_state[k]
        if v == 'none':
            turn_dialog_state.pop(k)
            continue
        vv = last_dialog_state.get(k)
        try:
            idx = slot_meta.index(k)
            if vv != v:
                if v == 'dontcare' and OP_SET[op_code].get('dontcare') is not None:
                    op_labels[idx] = 'dontcare'
                elif v == 'yes' and OP_SET[op_code].get('yes') is not None:
                    op_labels[idx] = 'yes'
                elif v == 'no' and OP_SET[op_code].get('no') is not None:
                    op_labels[idx] = 'no'
                else:
                    op_labels[idx] = 'update'
                    generate_y.append([tokenizer.tokenize(v) + ['[EOS]'], idx])
            elif vv == v:
                op_labels[idx] = 'carryover'
        except ValueError:
            continue

    for k, v in last_dialog_state.items():
        vv = turn_dialog_state.get(k)
        try:
            idx = slot_meta.index(k)
            if vv is None:
                if OP_SET[op_code].get('delete') is not None:
                    op_labels[idx] = 'delete'
                else:
                    op_labels[idx] = 'update'
                    generate_y.append([['[NULL]', '[EOS]'], idx])
        except ValueError:
            continue
    gold_state = [str(k) + '-' + str(v) for k, v in turn_dialog_state.items()]
    if len(generate_y) > 0:
        generate_y = sorted(generate_y, key=lambda lst: lst[1])
        generate_y, _ = [list(e) for e in list(zip(*generate_y))]

    if dynamic:
        op2id = OP_SET[op_code]
        generate_y = [tokenizer.convert_tokens_to_ids(y) for y in generate_y]
        op_labels = [op2id[i] for i in op_labels]

    return op_labels, generate_y, gold_state


#---------------------------------------------------------My Code----------------------------------------------------------------#

#----------------------------Instance---------------------------#
class TransDSTInstance:
    def __init__(self, ID,
                 turn_domain,
                 turn_id,
                 turn_utter,
                 dialog_history,
                 last_dialog_state,
                 generate_y,
                 gold_state,
                 max_seq_length,
                 slot_meta,
                 is_last_turn,
                 op_code='4'):
        self.id = ID
        self.turn_domain = turn_domain
        self.turn_id = turn_id
        self.turn_utter = turn_utter
        self.dialog_history = dialog_history
        self.corrupt_last_dialog_state = deepcopy(last_dialog_state)
        self.last_dialog_state = last_dialog_state
        self.gold_p_state = last_dialog_state
        self.generate_y = generate_y
        self.gold_state = gold_state
        self.max_seq_length = max_seq_length
        self.slot_meta = slot_meta
        self.is_last_turn = is_last_turn
        self.op2id = OP_SET[op_code]
        self.op_code = op_code

    def shuffle_state(self, rng, slot_meta=None):
        new_y = deepcopy(self.generate_y)
        if slot_meta is None:
            temp = list(zip(self.slot_meta, new_y))
            rng.shuffle(temp)
        else:
            #回到原始的顺序
            indices = list(range(len(slot_meta)))
            for idx, st in enumerate(slot_meta):
                indices[self.slot_meta.index(st)] = idx
            temp = list(zip(self.slot_meta, new_y, indices))
            temp = sorted(temp, key=lambda x: x[-1])
        temp = list(zip(*temp))
        self.slot_meta = list(temp[0])
        self.generate_y = list(temp[1])

    def make_instance(self, tokenizer, max_seq_length=None,
                      word_dropout=0., slot_token='[SLOT]', corrupt_method=None, corrupt_p=0.2, slot_values_lengths=None, slot_values_arr=None, slot_values_p=None):
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        state = []
        substitute_generate_y = {}
        for idx, s in enumerate(self.slot_meta):
            state.append(slot_token)
            k = s.split('-')
            v = self.last_dialog_state.get(s)
            # 在这里故意引入错误
            should_corrupt = corrupt_method is not None and np.random.binomial(1, corrupt_p) == 1
            if v is not None:
                k.extend(["-"])
                if should_corrupt:
                    # 先 tokenize
                    t = tokenizer.tokenize(' '.join(k))
                    if corrupt_method == "random":
                        # 得到随机 ID
                        random_value_ids = np.random.randint(1000, tokenizer.vocab_size, slot_values_lengths[s])
                        # 将 ID 转回 Token，拼回去
                        t.extend(tokenizer.convert_ids_to_tokens(random_value_ids))
                        # 应该生成原来的值
                        substitute_generate_y[idx] = ["[SLOT]"] + tokenizer.tokenize(v) + ["[EOS]"]
                    elif corrupt_method == "value":
                        # 按照概率选择一个值
                        vid = np.random.choice(len(slot_values_arr[s]), p=slot_values_p[s])
                        value_tokens = slot_values_arr[s][vid]
                        t.extend(value_tokens)
                        if tokenizer.convert_tokens_to_string(value_tokens) != v:
                            substitute_generate_y[idx] = ["[SLOT]"] + tokenizer.tokenize(v) + ["[EOS]"]
                else:
                    k.extend([v])
                    t = tokenizer.tokenize(' '.join(k))
            else:
                if should_corrupt:
                    k.extend(["-"])
                    # 先 tokenize
                    t = tokenizer.tokenize(' '.join(k))
                    if corrupt_method == "random":
                        # 得到随机 ID
                        random_value_ids = np.random.randint(1000, tokenizer.vocab_size, slot_values_lengths[s])
                        # 将 ID 转回 Token，拼回去
                        t.extend(tokenizer.convert_ids_to_tokens(random_value_ids))
                    elif corrupt_method == "value":
                        # 按照概率选择一个值
                        vid = np.random.choice(len(slot_values_arr[s]), p=slot_values_p[s])
                        value_tokens = slot_values_arr[s][vid]
                        t.extend(value_tokens)

                    # 模型有 Delete 应该生成 Delete，否则 update -> null
                    if OP_SET[self.op_code].get('delete') is not None:
                        substitute_generate_y[idx] = ["[SLOT]", "[Delete]", "[EOS]"]
                    else:
                        substitute_generate_y[idx] = ["[SLOT]", "[NULL]", "[EOS]"]
                else:
                    t = tokenizer.tokenize(' '.join(k))
                    t.extend(["-", "[NULL]"])
            state.extend(t)
        avail_length_1 = max_seq_length - len(state) - 3
        diag_1 = tokenizer.tokenize(self.dialog_history)
        diag_2 = tokenizer.tokenize(self.turn_utter)
        avail_length = avail_length_1 - len(diag_2)

        if len(diag_1) > avail_length:  # truncated
            avail_length = len(diag_1) - avail_length
            diag_1 = diag_1[avail_length:]

        if len(diag_1) == 0 and len(diag_2) > avail_length_1:
            avail_length = len(diag_2) - avail_length_1
            diag_2 = diag_2[avail_length:]

        drop_mask = [0] + [1] * len(diag_1) + [0] + [1] * len(diag_2) + [0]
        diag_1 = ["[CLS]"] + diag_1 + ["[SEP]"]
        diag_2 = diag_2 + ["[SEP]"]
        segment = [0] * len(diag_1) + [1] * len(diag_2)

        diag = diag_1 + diag_2
        # word dropout
        if word_dropout > 0.:
            drop_mask = np.array(drop_mask)
            word_drop = np.random.binomial(drop_mask.astype('int64'), word_dropout)
            diag = [w if word_drop[i] == 0 else '[UNK]' for i, w in enumerate(diag)]
        input_ = diag + state
        segment = segment + [1]*len(state)
        self.input_ = input_

        self.segment_id = segment
        slot_position = []
        for i, t in enumerate(self.input_):
            if t == slot_token:
                slot_position.append(i)
        self.slot_position = slot_position

        input_mask = [1] * len(self.input_)
        self.input_id = tokenizer.convert_tokens_to_ids(self.input_)
        if len(input_mask) < max_seq_length:
            self.input_id = self.input_id + [0] * (max_seq_length-len(input_mask))
            self.segment_id = self.segment_id + [0] * (max_seq_length-len(input_mask))
            input_mask = input_mask + [0] * (max_seq_length-len(input_mask))

        self.input_mask = input_mask
        self.domain_id = domain2id[self.turn_domain]
        self.generate_ids = [tokenizer.convert_tokens_to_ids(substitute_generate_y.get(idx, y)) for idx, y in enumerate(self.generate_y)]


Instance = {
    'TransDST':TransDSTInstance,
    # 'CompactTransDST':CompactTransDSTInstance,
    'TransDSTV2':TransDSTInstance,
    'TransDSTV3':TransDSTInstance,
}
#----------------------------Dataset----------------------------#
class TransDSTMultiWozDataset(Dataset):
    '''
        Input: 
                data, tokenizer, slot_meta, max_seq_length, rng,
                ontology, word_dropout=0.1
    '''
    def __init__(self, data, tokenizer, slot_meta, max_seq_length, rng,
                 ontology, word_dropout=0.1, shuffle_state=False, shuffle_p=0.0, corrupt_method=None, corrupt_p=0.2):
        self.data = data
        self.len = len(data)
        self.tokenizer = tokenizer
        self.slot_meta = slot_meta
        self.max_seq_length = max_seq_length
        self.ontology = ontology
        self.word_dropout = word_dropout
        self.shuffle_state = shuffle_state
        self.shuffle_p = shuffle_p
        self.rng = rng
        self.corrupt_p = corrupt_p
        self.corrupt_method = corrupt_method
        self._preprocess_slots()

    def _hash_generate_y(self, generate_y):
        return ':'.join(generate_y)

    def _preprocess_slots(self):
        self.slot_values = {k: dict() for k in self.slot_meta}
        self.slot_values_arr = {k: [] for k in self.slot_meta}
        self.slot_values_p = {k: [] for k in self.slot_meta}
        self.slot_lengths = {k: [0, 0] for k in self.slot_meta}
        for one in self.data:
            for slot_meta, generate_y in zip(one.slot_meta, one.generate_y):
                if "[" not in generate_y[1]:
                    self.slot_lengths[slot_meta][1] += 1
                    self.slot_lengths[slot_meta][0] += len(generate_y) - 2
                    d = self.slot_values[slot_meta].get(self._hash_generate_y(generate_y), {"text": None, "freq": 0})
                    d["text"] = generate_y
                    d["freq"] += 1
                    self.slot_values[slot_meta][self._hash_generate_y(generate_y)] = d
        for k in self.slot_lengths:
            self.slot_lengths[k] = math.ceil(self.slot_lengths[k][0] / self.slot_lengths[k][1])

        for k in self.slot_values:
            cnt = 0
            one_slot_values = self.slot_values[k]
            for hshkey in one_slot_values:
                d = one_slot_values[hshkey]
                self.slot_values_arr[k].append(d["text"][1:-1])
                self.slot_values_p[k].append(d["freq"])
                cnt += d["freq"]
            for i in range(len(self.slot_values_p[k])):
                self.slot_values_p[k][i] /= cnt

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.shuffle_state and self.shuffle_p > 0.:
            if self.rng.random() < self.shuffle_p:
                self.data[idx].shuffle_state(self.rng, None)
            else:
                self.data[idx].shuffle_state(self.rng, self.slot_meta)
        if self.word_dropout > 0 or self.shuffle_state or self.corrupt_method in ["value", "random"]:
            self.data[idx].make_instance(self.tokenizer,
                                         word_dropout=self.word_dropout, 
                                         corrupt_method=self.corrupt_method, 
                                         corrupt_p=self.corrupt_p, 
                                         slot_values_lengths=self.slot_lengths, 
                                         slot_values_arr=self.slot_values_arr, 
                                         slot_values_p=self.slot_values_p)
        return self.data[idx]

    def collate_fn(self, batch):
        input_ids = torch.tensor([f.input_id for f in batch], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in batch], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_id for f in batch], dtype=torch.long)
        state_position_ids = torch.tensor([f.slot_position for f in batch], dtype=torch.long)
        gen_ids = [b.generate_ids for b in batch]
        max_update = max([len(b) for b in gen_ids])
        max_value = max([len(b) for b in flatten(gen_ids)])
        for bid, b in enumerate(gen_ids):
            n_update = len(b)
            for idx, v in enumerate(b):
                b[idx] = v + [0] * (max_value - len(v))
            gen_ids[bid] = b + [[0] * max_value] * (max_update - n_update)
        gen_ids = torch.tensor(gen_ids, dtype=torch.long)

        return {'input_ids':input_ids, 
                'token_type_ids':segment_ids, 
                'attention_mask':input_mask, 
                'slot_positions':state_position_ids,
                'tgt_seq':gen_ids}


#----------------------------Method------------------------------#
def TransDSTpostprocessing(slot_meta, last_dialog_state, generated, tokenizer, op_code, gold_gen={}):
    gen_token = [tokenizer.convert_ids_to_tokens(generated[i]) for i in range(len(generated))]
    for idx, st in enumerate(slot_meta):
        if gen_token[idx][1] == '[DontCare]' and OP_SET[op_code].get('dontcare') is not None:
            last_dialog_state[st] = 'dontcare'
        elif gen_token[idx][1] == '[YES]' and OP_SET[op_code].get('yes') is not None:
            last_dialog_state[st] = 'yes'
        elif gen_token[idx][1] == '[NO]' and OP_SET[op_code].get('no') is not None:
            last_dialog_state[st] = 'no'
        elif gen_token[idx][1] == '[NULL]' and last_dialog_state.get(st):
            last_dialog_state.pop(st)
        elif gen_token[idx][1] == '[Delete]' and last_dialog_state.get(st) and OP_SET[op_code].get('delete') is not None:
            last_dialog_state.pop(st) 
        elif gen_token[idx][1] == '[CarryOver]' and OP_SET[op_code].get('carryover') is not None:
            continue
        else:
            g = gen_token[idx][1:]
            gen = []
            for gg in g:
                if gg == '[EOS]':
                    break
                gen.append(gg)
            gen = ' '.join(gen).replace(' ##', '')
            gen = gen.replace(' : ', ':').replace('##', '')
            if gold_gen and gold_gen.get(st) and gold_gen[st] not in ['dontcare']:
                gen = gold_gen[st]

            if gen == '[NULL]' and last_dialog_state.get(st):
                last_dialog_state.pop(st)
            else:
                last_dialog_state[st] = gen
    return generated, last_dialog_state


def TransDST_make_turn_label(slot_meta, last_dialog_state, turn_dialog_state,
                    tokenizer, op_code='4'):
    '''从last_dialog_state 和 turn_dialog_state 中推断出turn_label
        和som-dst的区别是: som-dst产生op state和generate_y两套监督信息。
                          TransDST则将op state和generate_y统合到一起。
        #------------------------------------------------------------------------------------------------#
        Ex: last_dialog_state: {"restaurant name": McDowbload, "restaurant address": None}

            turn_dialog_state: {"restaurant name": McDowbload, 
                                "restaurant address": None, 
                                "restaurant price":cheap, 
                                "restaurant food": fast food}
            som-dst:
                    op state:  {"restaurant name": carryover, 
                                "restaurant address": carryover, 
                                "restaurant price":update, 
                                "restaurant food": update}

                    generate_y: {"restaurant price":[cheap], "restaurant food": [fast,food]}
            TransDST:
                    generate_y: {"restaurant name": [carryover], 
                                "restaurant address": [carryover], 
                                "restaurant price":[cheap], 
                                "restaurant food": [fast,food]}
        #------------------------------------------------------------------------------------------------#
    '''
    generate_y = []
    keys = slot_meta
    for k in keys:
        v = turn_dialog_state.get(k)
        if v == 'none':
            v = None
            turn_dialog_state.pop(k)
        vv = last_dialog_state.get(k)
        try:
            idx = slot_meta.index(k)
            if vv != v:
                if v == 'dontcare' and OP_SET[op_code].get('dontcare') is not None:
                    generate_y.append([['[SLOT]'] + ['[DontCare]'] + ['[EOS]'], idx])
                elif v == 'yes' and OP_SET[op_code].get('yes') is not None:
                    generate_y.append([['[SLOT]'] + ['[YES]'] + ['[EOS]'], idx])
                elif v == 'no' and OP_SET[op_code].get('no') is not None:
                    generate_y.append([['[SLOT]'] + ['[NO]'] + ['[EOS]'], idx])
                elif v is None and vv is not None:
                    if OP_SET[op_code].get('delete') is not None:
                        generate_y.append([['[SLOT]'] + ['[Delete]'] + ['[EOS]'], idx])
                    else:
                        generate_y.append([['[SLOT]'] + ['[NULL]'] + ['[EOS]'], idx])
                else:
                    generate_y.append([['[SLOT]'] + tokenizer.tokenize(v) + ['[EOS]'], idx])
            elif vv == v:
                generate_y.append([['[SLOT]'] + ['[CarryOver]'] + ['[EOS]'], idx])

        except ValueError:
            continue

    gold_state = [str(k) + '-' + str(v) for k, v in turn_dialog_state.items()]
    if len(generate_y) > 0:
        generate_y = sorted(generate_y, key=lambda lst: lst[1])
        generate_y, _ = [list(e) for e in list(zip(*generate_y))]


    return generate_y, gold_state


def TransDST_prepare_dataset(data_path, tokenizer, slot_meta,
                    n_history, max_seq_length, diag_level=False, op_code='4', use_cache=True, args=None):
    if use_cache:
        if args and args.model == 'CompactTransDST':
            cache_file = data_path.replace('.json','_CompactTransDST{}.data'.format(op_code))
        else:
            cache_file = data_path.replace('.json','_TransDST{}.data'.format(op_code))
        if os.path.exists(cache_file):
            logging.info("Loading data from {}.........".format(cache_file))
            with open(cache_file,'rb') as f:
                data = pickle.load(f)
        else:
            logging.info("Encoding data into {}.........".format(cache_file))        
            dials = json.load(open(data_path))
            data = []
            domain_counter = {}
            max_resp_len, max_value_len = 0, 0
            max_line = None
            for dial_dict in tqdm(dials, mininterval=30):
                for domain in dial_dict["domains"]:
                    if domain not in EXPERIMENT_DOMAINS:
                        continue
                    if domain not in domain_counter.keys():
                        domain_counter[domain] = 0
                    domain_counter[domain] += 1

                dialog_history = []
                last_dialog_state = {}
                last_uttr = ""
                for ti, turn in enumerate(dial_dict["dialogue"]):
                    turn_domain = turn["domain"]
                    if turn_domain not in EXPERIMENT_DOMAINS:
                        continue
                    turn_id = turn["turn_idx"]
                    turn_uttr = (turn["system_transcript"] + ' ; ' + turn["transcript"]).strip()
                    dialog_history.append(last_uttr)
                    turn_dialog_state = fix_general_label_error(turn["belief_state"], False, slot_meta)
                    last_uttr = turn_uttr

                    generate_y, gold_state = TransDST_make_turn_label(slot_meta, last_dialog_state,
                                                                        turn_dialog_state,
                                                                        tokenizer, op_code)
                    if (ti + 1) == len(dial_dict["dialogue"]):
                        is_last_turn = True
                    else:
                        is_last_turn = False

                    instance = Instance[args.model](dial_dict["dialogue_idx"], 
                                                    turn_domain,
                                                    turn_id, turn_uttr, 
                                                    ' '.join(dialog_history[-n_history:]),
                                                    last_dialog_state, 
                                                    generate_y, 
                                                    gold_state, 
                                                    max_seq_length, 
                                                    slot_meta,
                                                    is_last_turn, 
                                                    op_code=op_code)
                    instance.make_instance(tokenizer)
                    data.append(instance)
                    last_dialog_state = turn_dialog_state
            
            with open(cache_file,'wb') as f:
                pickle.dump(data,f)
            
    return data

if __name__ == '__main__':
    from transformers import BertTokenizer
    ontology_data = '/data/lyh/DST/som-dst/data/mwz2.1/ontology.json'
    ontology = json.load(open(ontology_data))
    slot_meta, ontology = make_slot_meta(ontology)
    op2id = OP_SET['3-1']
    tokenizer = BertTokenizer("/data/lyh/DST/som-dst/assets/TransDST_vocab.txt", do_lower_case=True)
    
    s = 'i like eating [SLOT] and [PAD]'
    ss = s.split()
    ids = [tokenizer.convert_tokens_to_ids(si) for si in ss]

    print(tokenizer.decode(ids))
    train_data_raw = TransDST_prepare_dataset(data_path='/data/lyh/DST/som-dst/data/mwz2.1/train_dials.json',
                                     tokenizer=tokenizer,
                                     slot_meta=slot_meta,
                                     n_history=1,
                                     max_seq_length=256,
                                     op_code='3-1')

    dev_data_raw = TransDST_prepare_dataset(data_path='/data/lyh/DST/som-dst/data/mwz2.1/dev_dials.json',
                                     tokenizer=tokenizer,
                                     slot_meta=slot_meta,
                                     n_history=1,
                                     max_seq_length=256,
                                     op_code='3-1')
    
    test_data_raw = TransDST_prepare_dataset(data_path='/data/lyh/DST/som-dst/data/mwz2.1/test_dials.json',
                                     tokenizer=tokenizer,
                                     slot_meta=slot_meta,
                                     n_history=1,
                                     max_seq_length=256,
                                     op_code='3-1')
