"""
SOM-DST
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

from Model.model import *
from Model.transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertConfig
from utils.data_utils import prepare_dataset, MultiWozDataset
from utils.data_utils import make_slot_meta, domain2id, OP_SET, make_turn_label, postprocessing
from utils.eval_utils import compute_prf, compute_acc, per_domain_join_accuracy
from utils.ckpt_utils import download_ckpt, convert_ckpt_compatible
from evaluation import model_evaluation
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import argparse
import random
import os
import json
import time
from collections import OrderedDict
import logging
#set up log
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('./log/log_{}.json'.format(log_time))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    #初始化tensorboard
    logpath = args.save_dir+'/log/'
    if not os.path.exists(logpath):
        os.mkdir(logpath)
    writer = SummaryWriter(logpath)

    def worker_init_fn(worker_id):
        np.random.seed(args.random_seed + worker_id)
    n_gpu = 0
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    #固定随机种子
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    rng = random.Random(args.random_seed)
    torch.manual_seed(args.random_seed)

    ontology = json.load(open(args.ontology_data))
    slot_meta, ontology = make_slot_meta(ontology)
    op2id = OP_SET[args.op_code]
    logger.info(str(op2id))
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)

    train_data_raw = prepare_dataset(data_path=args.train_data_path,
                                     tokenizer=tokenizer,
                                     slot_meta=slot_meta,
                                     n_history=args.n_history,
                                     max_seq_length=args.max_seq_length,
                                     op_code=args.op_code)

    train_data = MultiWozDataset(train_data_raw,
                                 tokenizer,
                                 slot_meta,
                                 args.max_seq_length,
                                 rng,
                                 ontology,
                                 args.word_dropout,
                                 args.shuffle_state,
                                 args.shuffle_p)
    logger.info("# train examples {}".format(len(train_data_raw)))

    dev_data_raw = prepare_dataset(data_path=args.dev_data_path,
                                   tokenizer=tokenizer,
                                   slot_meta=slot_meta,
                                   n_history=args.n_history,
                                   max_seq_length=args.max_seq_length,
                                   op_code=args.op_code)
    logger.info("# dev examples {}".format(len(dev_data_raw)))

    test_data_raw = prepare_dataset(data_path=args.test_data_path,
                                    tokenizer=tokenizer,
                                    slot_meta=slot_meta,
                                    n_history=args.n_history,
                                    max_seq_length=args.max_seq_length,
                                    op_code=args.op_code)
    logger.info("# test examples {}".format(len(test_data_raw)))

    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.dropout = args.dropout
    model_config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
    model_config.hidden_dropout_prob = args.hidden_dropout_prob

    model = SomDST(bertpath = args.bert_ckpt_path,
                    config = model_config, 
                    n_op = len(op2id),
                    n_domain = len(domain2id), 
                    update_id = op2id['update'], 
                    exclude_domain= args.exclude_domain)
    # re-initialize added special tokens ([SLOT], [NULL], [EOS])
    model.encoder.bert.embeddings.word_embeddings.weight.data[1].normal_(mean=0.0, std=0.02)
    model.encoder.bert.embeddings.word_embeddings.weight.data[2].normal_(mean=0.0, std=0.02)
    model.encoder.bert.embeddings.word_embeddings.weight.data[3].normal_(mean=0.0, std=0.02)
    model.to(device)
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    enc_param_optimizer = list(model.encoder.named_parameters())
    enc_optimizer_grouped_parameters = [
        {'params': [p for n, p in enc_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in enc_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    num_train_steps = int(len(train_data_raw) / args.batch_size * args.n_epochs / args.grad_accumulation)
    enc_optimizer = AdamW(enc_optimizer_grouped_parameters, lr=args.enc_lr)
    enc_scheduler = get_linear_schedule_with_warmup(enc_optimizer, 
                                                    int(num_train_steps * args.enc_warmup),
                                                    num_training_steps=num_train_steps)
    dec_param_optimizer = list(model.decoder.parameters())
    dec_optimizer = AdamW(dec_param_optimizer, lr=args.dec_lr)
    dec_scheduler = get_linear_schedule_with_warmup(dec_optimizer,
                                                    int(num_train_steps * args.dec_warmup),
                                                    num_training_steps=num_train_steps)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  collate_fn=train_data.collate_fn,
                                  num_workers=args.num_workers,
                                  worker_init_fn=worker_init_fn)

    loss_fnc = nn.CrossEntropyLoss()
    best_score = {'epoch': 0, 'joint_acc': 0, 'op_acc': 0, 'final_slot_f1': 0}
    global_step = 0
    for epoch in range(args.n_epochs):
        batch_loss = []
        model.train()
        for step, batch in enumerate(train_dataloader):
            global_step += 1
            batch = [b.to(device) if not isinstance(b, int) else b for b in batch]
            input_ids, input_mask, segment_ids, state_position_ids, op_ids,\
            domain_ids, gen_ids, max_value, max_update = batch

            if rng.random() < args.decoder_teacher_forcing:  # teacher forcing
                teacher = gen_ids
            else:
                teacher = None

            loss_s, loss_g, loss_d = model(input_ids=input_ids,
                                            token_type_ids=segment_ids,
                                            state_positions=state_position_ids,
                                            attention_mask=input_mask,
                                            max_value=max_value,
                                            op_ids=op_ids,
                                            max_update=max_update,
                                            teacher=teacher,
                                            gen_ids=gen_ids,
                                            domain_ids=domain_ids)
            loss_s, loss_g, loss_d = loss_s.mean(), loss_g.mean(), loss_d.mean()
            loss = (loss_s + loss_g) / args.grad_accumulation
            if args.exclude_domain is not True:
                loss = loss + loss_d / args.grad_accumulation
            batch_loss.append(loss.item())

            loss.backward()
            if (step+1)%args.grad_accumulation == 0 or step == len(train_dataloader) - 1:
                enc_optimizer.step()
                enc_scheduler.step()
                dec_optimizer.step()
                dec_scheduler.step()
                model.zero_grad()
                

            if step % (100 * args.grad_accumulation) == 0:
                if args.exclude_domain is not True:
                    writer.add_scalars('loss',{'mean_loss': np.mean(batch_loss)*args.grad_accumulation,
                                                'state_loss': loss_s.item()*args.grad_accumulation,
                                                'gen_loss': loss_g.item(),
                                                'dom_loss': loss_d.item()}, global_step)

                    logger.info("[%d/%d] [%d/%d] mean_loss : %.3f, state_loss : %.3f, gen_loss : %.3f, dom_loss : %.3f" \
                          % (epoch+1, args.n_epochs, step,
                             len(train_dataloader), np.mean(batch_loss)*args.grad_accumulation,
                             loss_s.item()*args.grad_accumulation,
                             loss_g.item(), loss_d.item()))
                else:
                    logger.info("[%d/%d] [%d/%d] mean_loss : %.3f, state_loss : %.3f, gen_loss : %.3f" \
                          % (epoch+1, args.n_epochs, step,
                             len(train_dataloader), np.mean(batch_loss)*args.grad_accumulation,
                             loss_s.item()*args.grad_accumulation, loss_g.item()))
                batch_loss = []

        if (epoch+1) % args.eval_epoch == 0:
            eval_res = model_evaluation(model, dev_data_raw, tokenizer, slot_meta, epoch+1, args.op_code, args=args)
            if eval_res['joint_acc'] > best_score['joint_acc']:
                best_score = eval_res
                model_to_save = model.module if hasattr(model, 'module') else model
                save_path = os.path.join(args.save_dir, 'model_best.bin')
                torch.save(model_to_save.state_dict(), save_path)
            logger.info("Best Score : {}".format(best_score))
            logger.info("\n")

    logger.info("Test using best model...")
    best_epoch = best_score['epoch']
    ckpt_path = os.path.join(args.save_dir, 'model_best.bin')
    model = SomDST(model_config, len(op2id), len(domain2id), op2id['update'], args.exclude_domain)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(device)

    model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
                     is_gt_op=False, is_gt_p_state=False, is_gt_gen=False, args=args)
    model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
                     is_gt_op=False, is_gt_p_state=False, is_gt_gen=True, args=args)
    model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
                     is_gt_op=False, is_gt_p_state=True, is_gt_gen=False, args=args)
    model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
                     is_gt_op=False, is_gt_p_state=True, is_gt_gen=True, args=args)
    model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
                     is_gt_op=True, is_gt_p_state=False, is_gt_gen=False, args=args)
    model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
                     is_gt_op=True, is_gt_p_state=True, is_gt_gen=False, args=args)
    model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
                     is_gt_op=True, is_gt_p_state=False, is_gt_gen=True, args=args)
    model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
                     is_gt_op=True, is_gt_p_state=True, is_gt_gen=True, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_root", default='data/mwz2.1', type=str)
    parser.add_argument("--train_data", default='train_dials.json', type=str)
    parser.add_argument("--dev_data", default='dev_dials.json', type=str)
    parser.add_argument("--test_data", default='test_dials.json', type=str)
    parser.add_argument("--ontology_data", default='ontology.json', type=str)
    parser.add_argument("--vocab_path", default='assets/vocab.txt', type=str)
    parser.add_argument("--bert_config_path", default='assets/bert_config_base_uncased.json', type=str)
    parser.add_argument("--bert_ckpt_path", default='assets/pytorch_model.bin', type=str)
    parser.add_argument("--exp_dir", default='experiments', type=str)
    parser.add_argument("--mode",default="train",type=str)

    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--grad_accumulation", default=2, type=int)

    parser.add_argument("--enc_warmup", default=0.1, type=float)
    parser.add_argument("--dec_warmup", default=0.1, type=float)
    parser.add_argument("--enc_lr", default=4e-5, type=float)
    parser.add_argument("--dec_lr", default=1e-4, type=float)
    parser.add_argument("--n_epochs", default=30, type=int)
    parser.add_argument("--eval_epoch", default=1, type=int)

    parser.add_argument("--op_code", default="4", type=str)
    parser.add_argument("--slot_token", default="[SLOT]", type=str)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float)
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float)
    parser.add_argument("--decoder_teacher_forcing", default=0.5, type=float)
    parser.add_argument("--word_dropout", default=0.1, type=float)
    parser.add_argument("--not_shuffle_state", default=False, action='store_true')
    parser.add_argument("--shuffle_p", default=0.5, type=float)

    parser.add_argument("--n_history", default=1, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--msg", default=None, type=str)
    parser.add_argument("--exclude_domain", default=False, action='store_true')

    args = parser.parse_args()
    args.train_data_path = os.path.join(args.data_root, args.train_data)
    args.dev_data_path = os.path.join(args.data_root, args.dev_data)
    args.test_data_path = os.path.join(args.data_root, args.test_data)
    args.ontology_data = os.path.join(args.data_root, args.ontology_data)
    args.shuffle_state = False if args.not_shuffle_state else True
    if args.mode == 'train':
        args.save_dir = os.path.join(args.exp_dir,'exp_{}_bs{}_ga{}_lr{}_sd{}'.format(log_time, args.batch_size, 
                                                                                    args.grad_accumulation, args.enc_lr, args.random_seed))
    logger.info('pytorch version: {}'.format(torch.__version__))
    logger.info(args)
    main(args)
