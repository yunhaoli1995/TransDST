"""
SOM-DST
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

from utils.data_utils import TransDST_prepare_dataset, TransDSTMultiWozDataset, TransDSTInstance
from utils.data_utils import make_slot_meta, domain2id, OP_SET, make_turn_label, postprocessing, TransDSTpostprocessing
from utils.eval_utils import compute_prf, compute_acc, per_domain_join_accuracy
from Model.transformers import BertTokenizer, BertConfig
from Model.TransformerDST import TransformerDST

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import random
import numpy as np
import os
import time
import argparse
import json
from tqdm import tqdm
from copy import deepcopy
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpu = 0

if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()

def main(args):
    ontology = json.load(open(os.path.join(args.data_root, args.ontology_data)))
    slot_meta, _ = make_slot_meta(ontology)
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)
    data = TransDST_prepare_dataset(os.path.join(args.data_root, args.test_data),
                                    tokenizer=tokenizer,
                                    slot_meta=slot_meta,
                                    n_history=args.n_history,
                                    max_seq_length=args.max_seq_length,
                                    op_code=args.op_code)

    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.dropout = 0.1
    model_config.num_decoder_layers = 2
    model_config.pad_idx = 0
    model_config.decoder_dropout = 0.1
    model_config.eos_idx = 3
    op2id = OP_SET[args.op_code]
    model = TransformerDST(args.bert_ckpt_path, model_config)
    
    ckpt = torch.load(args.model_ckpt_path,map_location='cpu')
    model.load_state_dict(ckpt)
    for name,parameter in model.state_dict().items():
        f = (parameter == ckpt[name]).view(-1).numpy().tolist()
        if len(set(f)) != 1 and True not in f:
            print(name)

    model.eval()
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    if args.eval_all:
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         False, False, False)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         False, False, True)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         False, True, False)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         False, True, True)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         True, False, False)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         True, True, False)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         True, False, True)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         True, True, True)
    else:
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         args.gt_op, args.gt_p_state, args.gt_gen, args)


def model_evaluation(model, test_data, tokenizer, slot_meta, epoch, op_code='4',
                     is_gt_op=False, is_gt_p_state=False, is_gt_gen=False, args=None, mode='Val'):
    model.eval()
    op2id = OP_SET[op_code]
    id2op = {v: k for k, v in op2id.items()}
    id2domain = {v: k for k, v in domain2id.items()}

    slot_turn_acc, joint_acc, slot_F1_pred, slot_F1_count = 0, 0, 0, 0
    final_joint_acc, final_count, final_slot_F1_pred, final_slot_F1_count = 0, 0, 0, 0
    op_acc, op_F1, op_F1_count = 0, {k: 0 for k in op2id}, {k: 0 for k in op2id}
    all_op_F1_count = {k: 0 for k in op2id}

    tp_dic = {k: 0 for k in op2id}
    fn_dic = {k: 0 for k in op2id}
    fp_dic = {k: 0 for k in op2id}

    results = {}
    last_dialog_state = {}
    wall_times = []
    for i in tqdm(test_data,mininterval=120):
        if i.turn_id == 0:
            last_dialog_state = {}

        if is_gt_p_state is False:
            i.last_dialog_state = deepcopy(last_dialog_state)
            i.make_instance(tokenizer, word_dropout=0.)
        else:  # ground-truth previous dialogue state
            last_dialog_state = deepcopy(i.gold_p_state)
            i.last_dialog_state = deepcopy(last_dialog_state)
            i.make_instance(tokenizer, word_dropout=0.)

        input_ids = torch.LongTensor([i.input_id]).to(device)
        input_mask = torch.FloatTensor([i.input_mask]).to(device)
        segment_ids = torch.LongTensor([i.segment_id]).to(device)
        state_position_ids = torch.LongTensor([i.slot_position]).to(device)

        d_gold_op, _, _ = make_turn_label(slot_meta, last_dialog_state, i.gold_state,
                                          tokenizer, op_code, dynamic=True)
        gold_op_ids = torch.LongTensor([d_gold_op]).to(device)

        start = time.perf_counter()
        MAX_LENGTH = 10
        with torch.no_grad():
            if not hasattr(model, 'module'):
                g = model.generate(input_ids=input_ids,
                                token_type_ids=segment_ids,
                                attention_mask=input_mask,
                                slot_positions=state_position_ids,
                                MAX_LENGTH=MAX_LENGTH)
            else:
                g = model.module.generate(input_ids=input_ids,
                                token_type_ids=segment_ids,
                                attention_mask=input_mask,
                                slot_positions=state_position_ids,
                                MAX_LENGTH=MAX_LENGTH)
        if g.size(1) > 0:
            generated = g.tolist()
        else:
            generated = []

        if is_gt_gen:
            # ground_truth generation
            gold_gen = {'-'.join(ii.split('-')[:2]): ii.split('-')[-1] for ii in i.gold_state}
        else:
            gold_gen = {}
        generated, last_dialog_state = TransDSTpostprocessing(slot_meta, 
                                                            last_dialog_state,
                                                            generated[0], 
                                                            tokenizer, 
                                                            op_code, 
                                                            gold_gen)
        end = time.perf_counter()
        wall_times.append(end - start)
        pred_state = []
        for k, v in last_dialog_state.items():
            pred_state.append('-'.join([k, v]))

        if set(pred_state) == set(i.gold_state):
            joint_acc += 1
        key = str(i.id) + '_' + str(i.turn_id)
        results[key] = [pred_state, i.gold_state]

        # Compute prediction slot accuracy
        temp_acc = compute_acc(set(i.gold_state), set(pred_state), slot_meta)
        slot_turn_acc += temp_acc

        # Compute prediction F1 score
        temp_f1, temp_r, temp_p, count = compute_prf(i.gold_state, pred_state)
        slot_F1_pred += temp_f1
        slot_F1_count += count

        if i.is_last_turn:
            final_count += 1
            if set(pred_state) == set(i.gold_state):
                final_joint_acc += 1
            final_slot_F1_pred += temp_f1
            final_slot_F1_count += count

    joint_acc_score = joint_acc / len(test_data)
    turn_acc_score = slot_turn_acc / len(test_data)
    slot_F1_score = slot_F1_pred / slot_F1_count
    final_joint_acc_score = final_joint_acc / final_count
    final_slot_F1_score = final_slot_F1_pred / final_slot_F1_count
    latency = np.mean(wall_times) * 1000


    logging.info("------------------------------")
    logging.info('op_code: %s, is_gt_op: %s, is_gt_p_state: %s, is_gt_gen: %s' % \
          (op_code, str(is_gt_op), str(is_gt_p_state), str(is_gt_gen)))
    logging.info("Epoch {} joint accuracy : {:.4f}".format(epoch, joint_acc_score))
    logging.info("Epoch {} slot turn accuracy : {:.4f}".format(epoch, turn_acc_score))
    logging.info("Epoch {} slot turn F1: {:.4f}".format(epoch, slot_F1_score))
    logging.info("Final Joint Accuracy : {:.4f}".format(final_joint_acc_score))
    logging.info("Final slot turn F1 : {:.4f}".format(final_slot_F1_score))
    logging.info("Latency Per Prediction : %f ms" % latency)
    logging.info("-----------------------------\n")
    if mode == 'Test':
        path = os.path.join(args.save_dir, '{}_preds_{}.json'.format('Test',epoch))
    else:
        path = os.path.join(args.save_dir, 'preds_%d.json' % epoch)
    json.dump(results, open(path,'w'), indent=2)
    per_domain_join_accuracy(results, slot_meta)

    scores = {
                'epoch': epoch, 
                'joint_acc': joint_acc_score,
                'slot_acc': turn_acc_score, 
                'slot_f1': slot_F1_score,
                'final_slot_f1': final_slot_F1_score,
            }
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default='data/mwz2.1', type=str)
    parser.add_argument("--test_data", default='dev_dials.json', type=str)
    parser.add_argument("--ontology_data", default='ontology.json', type=str)
    parser.add_argument("--vocab_path", default='assets/TransDST_vocab.txt', type=str)
    parser.add_argument("--bert_config_path", default='assets/bert_config_base_uncased.json', type=str)
    parser.add_argument("--bert_ckpt_path", default='assets/pytorch_model.bin', type=str)
    parser.add_argument("--model_ckpt_path", default='experiments/TransDST_04-20-19-10_bs32_declayers2_enclr4e-05_declr0.0001_sd42/model_best.bin', type=str)
    parser.add_argument("--save_dir", default='experiments/TransDST_04-20-19-10_bs32_declayers2_enclr4e-05_declr0.0001_sd42/', type=str)
    parser.add_argument("--n_history", default=1, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--op_code", default="4", type=str)

    parser.add_argument("--gt_op", default=False, action='store_true')
    parser.add_argument("--gt_p_state", default=False, action='store_true')
    parser.add_argument("--gt_gen", default=False, action='store_true')
    parser.add_argument("--eval_all", default=False, action='store_true')

    args = parser.parse_args()
    
    main(args)
