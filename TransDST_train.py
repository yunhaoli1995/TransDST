"""
SOM-DST
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
from Model.TransformerDST import TransformerDST, TransformerDSTV2, TransformerDSTV3, CompactTransformerDST
from Model.transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertConfig
from utils.data_utils import *
from TransDST_evaluation import model_evaluation
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

MODEL_DICT = {
    'TransDST':TransformerDST,
    'CompactTransDST':CompactTransformerDST,
    'TransDSTV2':TransformerDSTV2,
    'TransDSTV3':TransformerDSTV3,
}
Dataset = {
    'TransDST':TransDSTMultiWozDataset,
    'CompactTransDST':CompactTransDSTMultiWozDataset,
    'TransDSTV2':TransDSTMultiWozDataset,
    'TransDSTV3':TransDSTMultiWozDataset,
}
#set up log
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
log_time = time.strftime("%m-%d-%H-%M", time.localtime())
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('./log/log_{}.json'.format(log_time))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_batch(batch, tokenizer):
    
    for i in range(min(4, batch['input_ids'].size()[0])):
        #input_ids
        logger.info("Input ids: {}".format(str(batch['input_ids'][i].numpy().tolist())))
        logging.info("Input sequence: {}".format(tokenizer.decode(batch['input_ids'][i])))
        # for j in range(30):
        #     logger.info("[SLOT]_{}: {}".format(j,tokenizer.decode(batch['tgt_seq'][i][j])))

def main(args):
    # 初始化tensorboard
    logpath = os.path.join(args.save_dir,'log')
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
    # 初始化并固定固定随机种子
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    rng = random.Random(args.random_seed)
    torch.manual_seed(args.random_seed)
    #------------------------------------------------------------Prepare data-----------------------------------------------------------#
    ontology = json.load(open(args.ontology_data))
    slot_meta, ontology = make_slot_meta(ontology)
    op2id = OP_SET[args.op_code]
    logger.info(str(op2id))
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)

    train_data_raw = TransDST_prepare_dataset(data_path=args.train_data_path,
                                            tokenizer=tokenizer,
                                            slot_meta=slot_meta,
                                            n_history=args.n_history,
                                            max_seq_length=args.max_seq_length,
                                            op_code=args.op_code,
                                            args=args)

    train_data = Dataset[args.model](train_data_raw,
                                    tokenizer,
                                    slot_meta,
                                    args.max_seq_length,
                                    rng,
                                    ontology,
                                    args.word_dropout,
                                    args.shuffle_state,
                                    args.shuffle_p)

    logger.info("# train examples {}".format(len(train_data_raw)))

    dev_data_raw = TransDST_prepare_dataset(data_path=args.dev_data_path,
                                            tokenizer=tokenizer,
                                            slot_meta=slot_meta,
                                            n_history=args.n_history,
                                            max_seq_length=args.max_seq_length,
                                            op_code=args.op_code,
                                            args=args)
    logger.info("# dev examples {}".format(len(dev_data_raw)))

    test_data_raw = TransDST_prepare_dataset(data_path=args.test_data_path,
                                            tokenizer=tokenizer,
                                            slot_meta=slot_meta,
                                            n_history=args.n_history,
                                            max_seq_length=args.max_seq_length,
                                            op_code=args.op_code,
                                            args=args)
    logger.info("# test examples {}".format(len(test_data_raw)))
    #-----------------------------------------------------------Prepare Model-----------------------------------------------------------#
    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.num_decoder_layers = args.num_decoder_layers
    model_config.pad_idx = 0
    model_config.eos_idx = 3
    model_config.decoder_dropout = args.decoder_dropout

    model = MODEL_DICT[args.model](args.bert_ckpt_path, model_config)
    # re-initialize added special tokens ([SLOT], [NULL], [EOS], [CarryOver], [DontCare], [YES], [NO])
    for i in range(1,8):
        model.encoder.embeddings.word_embeddings.weight.data[i].normal_(mean=0.0, std=0.02)
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
    #-----------------------------------------------------------Start Training-----------------------------------------------------------#
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  collate_fn=train_data.collate_fn,
                                  num_workers=args.num_workers,
                                  worker_init_fn=worker_init_fn)
    best_score = {'epoch': 0, 'joint_acc': 0, 'op_acc': 0, 'final_slot_f1': 0}
    global_step = 0
    print_batch_count = 1
    for epoch in range(args.n_epochs):
        batch_loss = []
        model.train()
        for step, batch in enumerate(train_dataloader):
            global_step += 1
            if print_batch_count > 0:
                print_batch(batch, tokenizer)
                print_batch_count -= 1

            batch = {k:batch[k].to(device) for k in batch}

            loss = model(**batch)
            
            loss = loss.mean()
            loss = loss / args.grad_accumulation
            batch_loss.append(loss.item())

            loss.backward()
            
            if (step+1)%args.grad_accumulation == 0 or step == len(train_dataloader) - 1:
                enc_optimizer.step()
                enc_scheduler.step()
                dec_optimizer.step()
                dec_scheduler.step()
                model.zero_grad()

            if step % (200 * args.grad_accumulation) == 0:
                local_loss = np.mean(batch_loss)*args.grad_accumulation
                writer.add_scalars('loss',{'mean_loss': local_loss})
                logger.info("[%d/%d] [%d/%d] mean_loss : %.5f"%(epoch+1, 
                                                                args.n_epochs, 
                                                                step, 
                                                                len(train_dataloader), 
                                                                local_loss))
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
    #---------------------------------------------------------Start Final Evaluation---------------------------------------------------------#

    logger.info("Test using best model...")
    best_epoch = best_score['epoch']
    model = MODEL_DICT[args.model](args.bert_ckpt_path, model_config)
    ckpt_path = os.path.join(args.save_dir, 'model_best.bin')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(device)

    model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code, args = args, mode='Test')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    # Path
    parser.add_argument("--data_root", default='data/mwz2.1', type=str)
    parser.add_argument("--train_data", default='train_dials.json', type=str)
    parser.add_argument("--dev_data", default='dev_dials.json', type=str)
    parser.add_argument("--test_data", default='test_dials.json', type=str)
    parser.add_argument("--ontology_data", default='ontology.json', type=str)
    parser.add_argument("--vocab_path", default='assets/TransDST_vocab.txt', type=str)
    parser.add_argument("--bert_config_path", default='assets/bert_config_base_uncased.json', type=str)
    parser.add_argument("--bert_ckpt_path", default='assets/pytorch_model.bin', type=str)
    parser.add_argument("--exp_dir", default='experiments', type=str)

    parser.add_argument("--mode",default="train",type=str)
    parser.add_argument("--model",default="CompactTransDST",type=str,
                        help="Model type, include: [TransDST, CompactTransDST,TransDSTV2, TransDSTV3]")
    # Model architecture
    parser.add_argument("--num_decoder_layers", default=2, type=int, help="Layers of transformer decoder")
    parser.add_argument("--decoder_dropout", default=0.1, type=float, help="Dropout rate of transformer decoder")

    # Training parameters
    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--grad_accumulation", default=1, type=int)

    parser.add_argument("--enc_warmup", default=0.1, type=float)
    parser.add_argument("--dec_warmup", default=0.1, type=float)
    parser.add_argument("--enc_lr", default=4e-5, type=float)
    parser.add_argument("--dec_lr", default=1e-4, type=float)
    parser.add_argument("--n_epochs", default=30, type=int)
    parser.add_argument("--eval_epoch", default=1, type=int)

    parser.add_argument("--op_code", default="2", type=str)
    parser.add_argument("--slot_token", default="[SLOT]", type=str)
    parser.add_argument("--not_shuffle_state", default=False, action='store_true')
    parser.add_argument("--shuffle_p", default=0.5, type=float)
    parser.add_argument("--word_dropout", default=0.1, type=float)

    parser.add_argument("--n_history", default=1, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)

    args = parser.parse_args()
    args.train_data_path = os.path.join(args.data_root, args.train_data)
    args.dev_data_path = os.path.join(args.data_root, args.dev_data)
    args.test_data_path = os.path.join(args.data_root, args.test_data)
    args.ontology_data = os.path.join(args.data_root, args.ontology_data)
    args.shuffle_state = False if args.not_shuffle_state else True

    args.save_dir = os.path.join(args.exp_dir,
                                '{}_{}_bs{}_declayers{}_enclr{}_declr{}_sd{}'.\
                                    format(args.model,
                                            log_time, 
                                            args.batch_size, 
                                            args.num_decoder_layers,
                                            args.enc_lr, 
                                            args.dec_lr,
                                            args.random_seed))
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    logger.info('pytorch version: {}'.format(torch.__version__))
    logger.info(args)
    main(args)
