import pickle
import json
import torch
import torch.nn as nn
import os
from Model.transformers import BertTokenizer

def error():
    '''
    统计错误样本
    '''
    tokenizer = BertTokenizer.from_pretrained('assets/')
    pred = json.load(open('experiments/checkpoint/pred/preds_0.json','r'))
    dials = pickle.load(open("data/mwz2.1/test_dials.data","rb"))

    error = {}
    for p, d in zip(pred, dials):
        if(set(pred[p][0]) != set(pred[p][1])):
            e = {}
            dial_id = p.split('_')[0]
            if dial_id not in error:
                error[dial_id] = []
            e['pred'] = pred[p][0]
            e['gold'] = pred[p][1]
            e['input'] = tokenizer.decode(d.input_id).replace('[PAD]','').split('[SLOT]')[0]
            e['turn_num'] = p.split('_')[1]
            error[dial_id].append(e)

    json.dump(error, open('error.json','w'), indent=2)


def f(a,b,c,**args):
    return a+b+c
c = f(**{'a':1,'b':2,'c':4,'d':4,'e':6})
b = 1
# error()
# tokenizer = BertTokenizer.from_pretrained('assets/')
# c =1

# a = nn.TransformerDecoderLayer