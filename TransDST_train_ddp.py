"""
SOM-DST
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
from Model.TransformerDST import TransformerDST
from Model.transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertConfig
from utils.data_utils import TransDST_prepare_dataset, TransDSTMultiWozDataset, TransDSTInstance
from utils.data_utils import make_slot_meta, domain2id, OP_SET
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

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def main(args):
    device = torch.device("cuda", args.local_rank)

    if dist.get_rank() == 0:
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
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    rng = random.Random(args.random_seed)
    torch.manual_seed(args.random_seed)

    if n_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    #------------------------------------------------------------Prepare data-----------------------------------------------------------#
    ontology = json.load(open(args.ontology_data))
    slot_meta, ontology = make_slot_meta(ontology)
    op2id = OP_SET[args.op_code]
    if dist.get_rank() == 0:
        logger.info(str(op2id))
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)

    train_data_raw = TransDST_prepare_dataset(data_path=args.train_data_path,
                                            tokenizer=tokenizer,
                                            slot_meta=slot_meta,
                                            n_history=args.n_history,
                                            max_seq_length=args.max_seq_length,
                                            op_code=args.op_code)

    train_data = TransDSTMultiWozDataset(train_data_raw,
                                        tokenizer,
                                        slot_meta,
                                        args.max_seq_length,
                                        rng,
                                        ontology,
                                        args.word_dropout)
    if dist.get_rank() == 0:                             
        logger.info("# train examples {}".format(len(train_data_raw)))

    dev_data_raw = TransDST_prepare_dataset(data_path=args.dev_data_path,
                                            tokenizer=tokenizer,
                                            slot_meta=slot_meta,
                                            n_history=args.n_history,
                                            max_seq_length=args.max_seq_length,
                                            op_code=args.op_code)
    if dist.get_rank() == 0:
        logger.info("# dev examples {}".format(len(dev_data_raw)))

    test_data_raw = TransDST_prepare_dataset(data_path=args.test_data_path,
                                            tokenizer=tokenizer,
                                            slot_meta=slot_meta,
                                            n_history=args.n_history,
                                            max_seq_length=args.max_seq_length,
                                            op_code=args.op_code)
    if dist.get_rank() == 0:
        logger.info("# test examples {}".format(len(test_data_raw)))
    #-----------------------------------------------------------Prepare Model-----------------------------------------------------------#
    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.num_decoder_layers = args.num_decoder_layers
    model_config.pad_idx = 0
    model_config.decoder_dropout = args.decoder_dropout

    model = TransformerDST(args.bert_ckpt_path, model_config)
    # re-initialize added special tokens ([SLOT], [NULL], [EOS], [CarryOver], [DontCare], [YES], [NO])
    if dist.get_rank() == 0:
        for i in range(1,8):
            model.encoder.embeddings.word_embeddings.weight.data[i].normal_(mean=0.0, std=0.02)
    
    model.to(args.local_rank)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    num_train_steps = int(len(train_data_raw) / args.batch_size * args.n_epochs /args.grad_accumulation)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    enc_param_optimizer = list(model.module.encoder.named_parameters())
    enc_optimizer_grouped_parameters = [
        {'params': [p for n, p in enc_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in enc_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    enc_optimizer = AdamW(enc_optimizer_grouped_parameters, lr=args.enc_lr)
    enc_scheduler = get_linear_schedule_with_warmup(enc_optimizer, 
                                                    int(num_train_steps * args.enc_warmup),
                                                    num_training_steps=num_train_steps)
    dec_param_optimizer = list(model.module.decoder.parameters())
    dec_optimizer = AdamW(dec_param_optimizer, lr=args.dec_lr)
    dec_scheduler = get_linear_schedule_with_warmup(dec_optimizer, 
                                                    int(num_train_steps * args.dec_warmup),
                                                    num_training_steps=num_train_steps)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  collate_fn=train_data.collate_fn,
                                  num_workers=args.num_workers,
                                  worker_init_fn=worker_init_fn)

    best_score = {'epoch': 0, 'joint_acc': 0, 'op_acc': 0, 'final_slot_f1': 0}
    for epoch in range(args.n_epochs):
        #DDP
        train_dataloader.sampler.set_epoch(epoch)
        epoch_loss = 0
        batch_loss = []
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = [b.to(args.local_rank) if not isinstance(b, int) else b for b in batch]
            input_ids, input_mask, segment_ids, state_position_ids, gen_ids = batch

            loss = model(input_ids=input_ids, 
                        token_type_ids=segment_ids, 
                        attention_mask=input_mask, 
                        slot_positions=state_position_ids,
                        tgt_seq=gen_ids)

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

            if step % (100 * args.grad_accumulation) == 0:
                if dist.get_rank() == 0:
                    local_loss = np.mean(batch_loss)*args.grad_accumulation
                    writer.add_scalars('loss',{'mean_loss': local_loss})
                    logger.info("[%d/%d] [%d/%d] mean_loss : %.4f"%(epoch+1, 
                                                                args.n_epochs, 
                                                                step, 
                                                                len(train_dataloader), 
                                                                local_loss))
                    batch_loss = []

        if (epoch+1) % args.eval_epoch == 0:
            if dist.get_rank() == 0:
                eval_res = model_evaluation(model, dev_data_raw, tokenizer, slot_meta, epoch+1, args.op_code)
                if eval_res['joint_acc'] > best_score['joint_acc']:
                    best_score = eval_res
                    save_path = os.path.join(args.save_dir,'Epoch%d_Loss%.2f_JA%.2f.bin'%(epoch,local_loss,eval_res))
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), save_path)
                logger.info("Best Score : {}".format(best_score))
                logger.info("\n")

    if dist.get_rank() == 0:
        logger.info("Test using best model...")
        best_epoch = best_score['epoch']
        ckpt_path = os.path.join(args.save_dir, 'model_best.bin')
        model = SomDST(model_config, len(op2id), len(domain2id), op2id['update'], args.exclude_domain)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt)
        model.to(device)

        model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
                        is_gt_op=False, is_gt_p_state=False, is_gt_gen=False)
        model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
                        is_gt_op=False, is_gt_p_state=False, is_gt_gen=True)
        model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
                        is_gt_op=False, is_gt_p_state=True, is_gt_gen=False)
        model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
                        is_gt_op=False, is_gt_p_state=True, is_gt_gen=True)
        model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
                        is_gt_op=True, is_gt_p_state=False, is_gt_gen=False)
        model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
                        is_gt_op=True, is_gt_p_state=True, is_gt_gen=False)
        model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
                        is_gt_op=True, is_gt_p_state=False, is_gt_gen=True)
        model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
                        is_gt_op=True, is_gt_p_state=True, is_gt_gen=True)


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
    parser.add_argument("--model",default="TransDST",type=str)

    # Model architecture
    parser.add_argument("--num_decoder_layers", default=3, type=int, help="Layers of transformer decoder")
    parser.add_argument("--decoder_dropout", default=0.1, type=float, help="Dropout rate of transformer decoder")

    # Training parameters
    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--grad_accumulation", default=1, type=int)

    parser.add_argument("--enc_warmup", default=0.1, type=float)
    parser.add_argument("--dec_warmup", default=0.1, type=float)
    parser.add_argument("--enc_lr", default=4e-5, type=float)
    parser.add_argument("--dec_lr", default=1e-4, type=float)
    parser.add_argument("--n_epochs", default=30, type=int)
    parser.add_argument("--eval_epoch", default=1, type=int)

    parser.add_argument("--op_code", default="4", type=str)
    parser.add_argument("--slot_token", default="[SLOT]", type=str)
    parser.add_argument("--word_dropout", default=0.1, type=float)

    parser.add_argument("--n_history", default=1, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)
    #Added by lyh
    parser.add_argument("--local_rank",default=-1, type=int)

    args = parser.parse_args()
    args.train_data_path = os.path.join(args.data_root, args.train_data)
    args.dev_data_path = os.path.join(args.data_root, args.dev_data)
    args.test_data_path = os.path.join(args.data_root, args.test_data)
    args.ontology_data = os.path.join(args.data_root, args.ontology_data)
    
    log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    args.save_dir = os.path.join(args.exp_dir,
                                'exp_{}_{}_bs{}_ga{}_lr{}_sd{}'.\
                                    format(args.model,
                                            log_time, 
                                            args.batch_size, 
                                            args.grad_accumulation,
                                            args.enc_lr, 
                                            args.random_seed))
    #DDP initialize
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
    #Initialize logger
    if dist.get_rank() == 0:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('./log/log_{}.json'.format(log_time))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info('pytorch version: {}'.format(torch.__version__))

        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        logger.info(args)

    main(args)

