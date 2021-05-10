"""
SOM-DST
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

import torch
import torch.nn as nn
from .transformers import BertPreTrainedModel, BertModel, GPT2LMHeadModel, BertConfig
from torch import Tensor
from .transformer import TransformerDecoder, TransformerDecoderV2, TransformerDecoderV3

class TransformerDST(nn.Module):
    '''
        我魔改的TransformerDST:先用bert对上下文X和对话状态[SLOT]_j slot-value进行双向编码
        然后对于第j个slot，取出对应的slot feature: h_{[SLOT]_j} 作为transformer decoder的第一个query feature: q0
        Input:
            input_ids: (B,Lk)
            token_type_ids: (B,Lk)
            attention_mask: (B,Lk)
            slot_positions: (B,J)
            tgt_seq: (B,J,Lq)
        Output:
            logits: (B,J,Lq,V)
            loss: (1)
    '''
    def __init__(self, bert_path, config, **args):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.encoder = BertModel.from_pretrained(bert_path, config=config)
        self.decoder = TransformerDecoder(config,
                                        self.encoder.embeddings.word_embeddings.weight, 
                                        self.encoder.embeddings.position_embeddings.weight)
    
    def encode(self, input_ids, token_type_ids, attention_mask):
        bert_outputs = self.encoder(input_ids=input_ids, 
                                    token_type_ids=token_type_ids, 
                                    attention_mask=attention_mask)
        enc_output = bert_outputs[:1][0]
        
        return enc_output
    
    def decode(self, tgt_seq, input_ids, enc_output, slot_positions):
        slot_pos = slot_positions[:, :, None].expand(-1, -1, enc_output.size(-1))
        slot_features = torch.gather(enc_output, 1, slot_pos)
        logits = self.decoder(tgt_seq, slot_features, input_ids, enc_output)
        
        return logits

    def forward(self, input_ids, token_type_ids, attention_mask, slot_positions, tgt_seq, **args):
        enc_output = self.encode(input_ids, token_type_ids, attention_mask)
        logits = self.decode(tgt_seq, input_ids, enc_output, slot_positions)
        # Compute loss
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = tgt_seq[..., 1:].contiguous()
        # Flatten the tokens
        loss = nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                                            shift_labels.view(-1),
                                            ignore_index=self.config.pad_idx)
        
        return loss


    def generate(self, input_ids, token_type_ids, attention_mask, slot_positions, MAX_LENGTH=12, **args):
        '''使用Greedy search生成
        '''
        enc_output = self.encode(input_ids, token_type_ids, attention_mask)
        B,Lk = input_ids.size()
        B,J =  slot_positions.size()          
        tgt_seq = torch.ones(B,J,1).to(input_ids.device).long()                     # (B,J,1)
        flag = torch.tensor([False]*J).to(input_ids.device)
        for i in range(MAX_LENGTH):
            logits = self.decode(tgt_seq, input_ids, enc_output, slot_positions)    # (B,J,Lq,V)
            _, w_idx = logits[...,-1,:].max(-1)                                     # (B,J)
            tgt_seq = torch.cat([tgt_seq, w_idx.unsqueeze(-1)], dim = -1)           # (B,J,Lq+1)
            flag = flag | (w_idx.squeeze(0) == self.config.eos_idx)
            if flag.sum() == J:                                                     #所有slot均解码完毕就break
                break
        return tgt_seq                                                              # (B,J,Lq_final)


class CompactTransformerDST(nn.Module):
    '''
        我魔改的TransformerDST:先用bert对上下文X和对话状态[SLOT]_j slot-value进行双向编码
        然后对于第j个slot，取出对应的slot feature: h_{[SLOT]_j} 作为transformer decoder的第一个query feature: q0
        将op和gen分开打包，以减少内存占用
        前向传播时，分别计算op和gen
        Input:
            input_ids: (B,Lk)
            token_type_ids: (B,Lk)
            attention_mask: (B,Lk)
            slot_positions: (B,J)
            tgt_seq: (B,J,Lq)
        Output:
            logits: (B,J,Lq,V)
            loss: (1)
    '''
    def __init__(self, bert_path, config, **args):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.encoder = BertModel.from_pretrained(bert_path, config=config)
        self.decoder = TransformerDecoder(config,
                                        self.encoder.embeddings.word_embeddings.weight, 
                                        self.encoder.embeddings.position_embeddings.weight)
    
    def prepare_input4decoder(self, enc_output, slot_positions, gen_idx, op_idx):
        B,Lk,H = enc_output.size()
        slot_pos = slot_positions[:, :, None].expand(-1, -1, H)
        slot_features = torch.gather(enc_output, 1, slot_pos).view(-1,H)

        gen_pos = gen_idx[:,None].expand(-1,H)
        op_pos = op_idx[:,None].expand(-1,H)

        gen_features = torch.gather(slot_features,0,gen_pos)
        op_features = torch.gather(slot_features,0,op_pos)

        return gen_features, op_features

    def ComputeLoss(logits,tgt_seq):
        # Compute loss
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = tgt_seq[..., 1:].contiguous()
        # Flatten the tokens
        loss = nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                                            shift_labels.view(-1),
                                            ignore_index=self.config.pad_idx)

        return loss

    def encode(self, input_ids, token_type_ids, attention_mask):
        bert_outputs = self.encoder(input_ids=input_ids, 
                                    token_type_ids=token_type_ids, 
                                    attention_mask=attention_mask)
        enc_output = bert_outputs[:1][0]
        
        return enc_output
    
    def decode(self, tgt_seq, input_ids, enc_output, slot_features):
        logits = self.decoder(tgt_seq, slot_features, input_ids, enc_output)
        
        return logits
 
    def forward(self, input_ids, token_type_ids, attention_mask, slot_positions, tgt_seq, op_ids, gen_idx, op_idx, **args):
        enc_output = self.encode(input_ids, token_type_ids, attention_mask)
        gen_features, op_features = self.prepare_input4decoder(enc_output, slot_positions, gen_idx, op_idx)
        logits1 = self.decode(tgt_seq, input_ids, enc_output, gen_features)
        logits2 = self.decode(op_ids, input_ids, enc_output, op_features)
        loss1 = self.ComputeLoss(tgt_seq, logits1)
        loss2 = self.ComputeLoss(op_ids, logits2)
        
        return (loss1+loss2)/2


    def generate(self, input_ids, token_type_ids, attention_mask, slot_positions, MAX_LENGTH=12, **args):
        '''使用Greedy search生成
        '''
        enc_output = self.encode(input_ids, token_type_ids, attention_mask)
        B,Lk = input_ids.size()
        B,J =  slot_positions.size()          
        tgt_seq = torch.ones(B,J,1).to(input_ids.device).long()                     # (B,J,1)
        flag = torch.tensor([False]*J).to(input_ids.device)
        for i in range(MAX_LENGTH):
            logits = self.decode(tgt_seq, input_ids, enc_output, slot_positions)    # (B,J,Lq,V)
            _, w_idx = logits[...,-1,:].max(-1)                                     # (B,J)
            tgt_seq = torch.cat([tgt_seq, w_idx.unsqueeze(-1)], dim = -1)           # (B,J,Lq+1)
            flag = flag | (w_idx.squeeze(0) == self.config.eos_idx)
            if flag.sum() == J:                                                     #所有slot均解码完毕就break
                break
        return tgt_seq


class TransformerDSTV2(nn.Module):
    '''
        我魔改的TransformerDST:先用bert对上下文X和对话状态[SLOT]_j slot-value进行双向编码
        然后对于第j个slot，取出对应的slot feature: h_{[SLOT]_j} 作为transformer decoder的第一个query feature: q0
        V2: Transformer decoder不会对contxt做cross attention
            +copy
        Input:
            input_ids: (B,Lk)
            token_type_ids: (B,Lk)
            attention_mask: (B,Lk)
            slot_positions: (B,J)
            tgt_seq: (B,J,Lq)
        Output:
            logits: (B,J,Lq,V)
            loss: (1)
    '''
    def __init__(self, bert_path, config, **args):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.encoder = BertModel.from_pretrained(bert_path, config=config)
        self.decoder = TransformerDecoderV2(config, 
                                        self.encoder.embeddings.word_embeddings.weight, 
                                        self.encoder.embeddings.position_embeddings.weight)
    
    def encode(self, input_ids, token_type_ids, attention_mask):
        bert_outputs = self.encoder(input_ids=input_ids, 
                                    token_type_ids=token_type_ids, 
                                    attention_mask=attention_mask)
        enc_output = bert_outputs[:1][0]
        
        return enc_output
    
    def decode(self, tgt_seq, input_ids, enc_output, slot_positions):
        slot_pos = slot_positions[:, :, None].expand(-1, -1, enc_output.size(-1))
        slot_features = torch.gather(enc_output, 1, slot_pos)
        logits = self.decoder(tgt_seq, slot_features, input_ids, enc_output)
        
        return logits

    def forward(self, input_ids, token_type_ids, attention_mask, slot_positions, tgt_seq, **args):
        enc_output = self.encode(input_ids, token_type_ids, attention_mask)
        prob = self.decode(tgt_seq, input_ids, enc_output, slot_positions)
        # Compute loss
        # Shift so that tokens < n predict n
        shift_logits = prob[..., :-1, :].contiguous()
        shift_labels = tgt_seq[..., 1:].contiguous()
        # Flatten the tokens
        loss = nn.functional.nll_loss(torch.log(shift_logits.view(-1, shift_logits.size(-1))), 
                                        shift_labels.view(-1),
                                        ignore_index=self.config.pad_idx)
        
        return loss


    def generate(self, input_ids, token_type_ids, attention_mask, slot_positions, MAX_LENGTH=12, **args):
        '''使用Greedy search生成
        '''
        enc_output = self.encode(input_ids, token_type_ids, attention_mask)
        B,Lk = input_ids.size()
        B,J =  slot_positions.size()          
        tgt_seq = torch.ones(B,J,1).to(input_ids.device).long()                     # (B,J,1)
        flag = torch.tensor([False]*J).to(input_ids.device)
        for i in range(MAX_LENGTH):
            logits = self.decode(tgt_seq, input_ids, enc_output, slot_positions)    # (B,J,Lq,V)
            _, w_idx = logits[...,-1,:].max(-1)                                     # (B,J)
            tgt_seq = torch.cat([tgt_seq, w_idx.unsqueeze(-1)], dim = -1)           # (B,J,Lq+1)
            flag = flag | (w_idx.squeeze(0) == self.config.eos_idx)
            if flag.sum() == J:                                                     #所有slot均解码完毕就break
                break
        return tgt_seq 


class TransformerDSTV3(nn.Module):
    '''
        我魔改的TransformerDST:先用bert对上下文X和对话状态[SLOT]_j slot-value进行双向编码
        然后对于第j个slot，取出对应的slot feature: h_{[SLOT]_j} 作为transformer decoder的第一个query feature: q0
        V3: +copy
        Input:
            input_ids: (B,Lk)
            token_type_ids: (B,Lk)
            attention_mask: (B,Lk)
            slot_positions: (B,J)
            tgt_seq: (B,J,Lq)
        Output:
            logits: (B,J,Lq,V)
            loss: (1)
    '''
    def __init__(self, bert_path, config, **args):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.encoder = BertModel.from_pretrained(bert_path, config=config)
        self.decoder = TransformerDecoderV3(config, 
                                        self.encoder.embeddings.word_embeddings.weight, 
                                        self.encoder.embeddings.position_embeddings.weight)
    
    def encode(self, input_ids, token_type_ids, attention_mask):
        bert_outputs = self.encoder(input_ids=input_ids, 
                                    token_type_ids=token_type_ids, 
                                    attention_mask=attention_mask)
        enc_output = bert_outputs[:1][0]
        
        return enc_output
    
    def decode(self, tgt_seq, input_ids, enc_output, slot_positions):
        slot_pos = slot_positions[:, :, None].expand(-1, -1, enc_output.size(-1))
        slot_features = torch.gather(enc_output, 1, slot_pos)
        logits = self.decoder(tgt_seq, slot_features, input_ids, enc_output)
        
        return logits

    def forward(self, input_ids, token_type_ids, attention_mask, slot_positions, tgt_seq, **args):
        enc_output = self.encode(input_ids, token_type_ids, attention_mask)
        prob = self.decode(tgt_seq, input_ids, enc_output, slot_positions)
        # Compute loss
        # Shift so that tokens < n predict n
        shift_logits = prob[..., :-1, :].contiguous()
        shift_labels = tgt_seq[..., 1:].contiguous()
        # Flatten the tokens
        loss = nn.functional.nll_loss(torch.log(shift_logits.view(-1, shift_logits.size(-1))), 
                                        shift_labels.view(-1),
                                        ignore_index=self.config.pad_idx)
        
        return loss


    def generate(self, input_ids, token_type_ids, attention_mask, slot_positions, MAX_LENGTH=12, **args):
        '''使用Greedy search生成
        '''
        enc_output = self.encode(input_ids, token_type_ids, attention_mask)
        B,Lk = input_ids.size()
        B,J =  slot_positions.size()          
        tgt_seq = torch.ones(B,J,1).to(input_ids.device).long()                     # (B,J,1)
        flag = torch.tensor([False]*J).to(input_ids.device)
        for i in range(MAX_LENGTH):
            logits = self.decode(tgt_seq, input_ids, enc_output, slot_positions)    # (B,J,Lq,V)
            _, w_idx = logits[...,-1,:].max(-1)                                     # (B,J)
            tgt_seq = torch.cat([tgt_seq, w_idx.unsqueeze(-1)], dim = -1)           # (B,J,Lq+1)
            flag = flag | (w_idx.squeeze(0) == self.config.eos_idx)
            if flag.sum() == J:                                                     #所有slot均解码完毕就break
                break
        return tgt_seq 

if __name__ == "__main__":
    path = '/data/lyh/DST/som-dst/assets/'
    bert_config = BertConfig.from_json_file('assets/bert_config_base_uncased.json')
    bert_config.num_decoder_layers = 3
    bert_config.pad_idx = 0
    bert_config.decoder_dropout = 0.2
    model = TransformerDSTV2(path, bert_config)
    
    input_ids = torch.tensor([[0,1,2,3,4,5,0,0,0]]*2)
    token_type_ids = torch.tensor([[1,1,1,1,1,1,0,0,0]]*2)
    attention_mask = torch.tensor([[1,1,1,1,1,1,0,0,0]]*2)
    slot_positions = torch.tensor([[0,2,4,5]]*2)
    tgt_seq = torch.tensor([[[1,2,3,0],[1,3,3,0],[1,4,5,6],[1,3,0,0]]]*2)

    logits, loss = model(input_ids, token_type_ids, attention_mask, slot_positions, tgt_seq)
    c = 1
    
