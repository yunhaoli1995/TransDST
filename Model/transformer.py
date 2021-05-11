import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

def get_non_pad_mask(seq, pad_idx):
    assert seq.dim() == 2
    return seq.ne(pad_idx).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q, pad_idx):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

class Transformer(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, n_src_vocab, len_max_seq, d_word_vec,
                n_layers, n_head, d_k, d_v,
                d_model, d_inner, embedding, dropout=0.1):

        super(Transformer, self).__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        #self.src_word_emb = nn.Embedding.from_pretrained(embedding, freeze=False)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, act_vocab_id):
        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq, pad_idx=self.pad_idx)
        non_pad_mask = get_non_pad_mask(src_seq, self.pad_idx)

        # -- Forward Word Embedding
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)
        # -- Forward Ontology Embedding
        ontology_embedding = self.src_word_emb(act_vocab_id)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        dot_prod = torch.sum(enc_output[:, :, None, :] * ontology_embedding[None, None, :, :], -1)
        #index = length[:, None, None].repeat(1, 1, dot_prod.size(-1))
        #pooled_dot_prod = dot_prod.gather(1, index).squeeze()
        pooled_dot_prod = dot_prod[:, 0, :]
        pooling_likelihood = torch.sigmoid(pooled_dot_prod)
        return pooling_likelihood, enc_output
 
class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn

class TransformerDecoder(nn.Module):
    ''' A decoder model with self attention mechanism.
        Input: 
            tgt_seq: Target sequence
            enc_output: Output of encoder
            config: Configuration of model, consists:
                    pad_idx: Pad index
                    num_decoder_layers: Layers of transformer decoder
                    decoder_dropout: Dropout rate of transformer decoder
    '''
    def __init__(self, config, bert_word_embedding_weights, bert_pos_embedding_weights):
        
        super(TransformerDecoder, self).__init__()
        
        self.vocab_size = config.vocab_size
        self.pad_idx = config.pad_idx
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        
        d_k = hidden_size // num_attention_heads
        d_v = hidden_size // num_attention_heads
        
        # -- word embedding layer
        self.tgt_word_emb = nn.Embedding(self.vocab_size, hidden_size, padding_idx=config.pad_idx)
        self.tgt_word_emb.weight = bert_word_embedding_weights
        # -- position embedding layer
        self.post_word_emb = nn.Embedding(self.vocab_size, hidden_size)  
        self.post_word_emb.weight = bert_pos_embedding_weights
        # -- Transformer decoder layer
        self.layer_stack = nn.ModuleList([
            DecoderLayer(hidden_size, 
                        config.intermediate_size, 
                        num_attention_heads,
                        d_k, d_v, dropout=config.decoder_dropout)
            ]*config.num_decoder_layers)
        

    def forward(self, tgt_seq, slot_features, src_seq, enc_output):
        '''tgt_seq: Concatenated slot values  (B,J,L)
            slot_features: Hidden features of slot (B,J,H)
            src_seq: Input sequence (B,Lk)
            enc_output: Sequence of encoded context from encoder: (B,Lk,H)
        '''
        # -- Reshape input
        B,J,Lq = tgt_seq.size()
        B,Lk,H = enc_output.size()
        tgt_seq = tgt_seq.contiguous().view(B*J,-1)                         #(B,J,Lq) -> (B*J,Lq)
        src_seq = src_seq.repeat(1,J).contiguous().view(B*J,-1)             #(B,Lk) -> (B*J,Lk)
        enc_output = enc_output.repeat(1,J,1).contiguous().view(B*J,Lk,-1)  #(B,Lk,H) -> (B*J,Lk,H)
        slot_features = slot_features.contiguous().view(B*J,1,-1)           #(B,J,H) -> (B*J,H)
        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq, self.pad_idx)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)                                                 #(B*J,Lq,Lq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq, pad_idx=self.pad_idx)    #(B*J,Lq,Lq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)             
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq, pad_idx=self.pad_idx)       #(B*J,Lq,Lk)

        # -- Forward
        max_len = tgt_seq.size()[1]
        position = torch.arange(0, max_len).long().unsqueeze(0).to(tgt_seq.device)
        dec_output = self.tgt_word_emb(tgt_seq) + self.post_word_emb(position)                              #(B*J,Lq,H)
        # -- Noting: 目标序列[SLOT]t1t2t3..[EOS]的embedding在进入decoder前 
        #            [SLOT]对应的feature要替换成为h_{[SLOT]_j} 因为h_{[SLOT]_j}
        #            和上下文通过self-attention进行了双向信息交互
        dec_output = torch.cat([slot_features, dec_output[:,1:]],dim=1)
        
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, enc_output,
                                                                non_pad_mask=non_pad_mask,
                                                                slf_attn_mask=slf_attn_mask,
                                                                dec_enc_attn_mask=dec_enc_attn_mask)        #(B*J,Lq,H)

        logits = torch.matmul(dec_output, self.tgt_word_emb.weight.transpose(0,1))                          #(B*J,Lq,H)*(H,V) -> (B*J,Lq,V)
        logits = logits.view(B,J,Lq,-1)                                                                     #(B*J,Lq,V) -> (B,J,Lq,V)

        return logits


class CompactTransformerDecoder(nn.Module):
    ''' A decoder model with self attention mechanism.
        Input: 
            tgt_seq: Target sequence
            enc_output: Output of encoder
            config: Configuration of model, consists:
                    pad_idx: Pad index
                    num_decoder_layers: Layers of transformer decoder
                    decoder_dropout: Dropout rate of transformer decoder
    '''
    def __init__(self, config, bert_word_embedding_weights, bert_pos_embedding_weights):
        
        super(CompactTransformerDecoder, self).__init__()
        
        self.vocab_size = config.vocab_size
        self.pad_idx = config.pad_idx
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        
        d_k = hidden_size // num_attention_heads
        d_v = hidden_size // num_attention_heads
        
        # -- word embedding layer
        self.tgt_word_emb = nn.Embedding(self.vocab_size, hidden_size, padding_idx=config.pad_idx)
        self.tgt_word_emb.weight = bert_word_embedding_weights
        # -- position embedding layer
        self.post_word_emb = nn.Embedding(self.vocab_size, hidden_size)  
        self.post_word_emb.weight = bert_pos_embedding_weights
        # -- Transformer decoder layer
        self.layer_stack = nn.ModuleList([
            DecoderLayer(hidden_size, 
                        config.intermediate_size, 
                        num_attention_heads,
                        d_k, d_v, dropout=config.decoder_dropout)
            ]*config.num_decoder_layers)
        

    def forward(self, tgt_seq, slot_features, src_seq, enc_output):
        '''tgt_seq: Concatenated slot values  (B,L)
            slot_features: Hidden features of slot (B,H)
            src_seq: Input sequence (B,Lk)
            enc_output: Sequence of encoded context from encoder: (B,Lk,H)
        '''
        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq, self.pad_idx)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)                                                 #(B,Lq,Lq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq, pad_idx=self.pad_idx)    #(B,Lq,Lq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)             
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq, pad_idx=self.pad_idx)       #(B,Lq,Lk)

        # -- Forward
        max_len = tgt_seq.size()[1]
        position = torch.arange(0, max_len).long().unsqueeze(0).to(tgt_seq.device)
        dec_output = self.tgt_word_emb(tgt_seq) + self.post_word_emb(position)                              #(B,Lq,H)
        # -- Noting: 目标序列[SLOT]t1t2t3..[EOS]的embedding在进入decoder前 
        #            [SLOT]对应的feature要替换成为h_{[SLOT]_j} 因为h_{[SLOT]_j}
        #            和上下文通过self-attention进行了双向信息交互
        dec_output = torch.cat([slot_features.unsqueeze(1), dec_output[:,1:]],dim=1)
        
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, enc_output,
                                                                non_pad_mask=non_pad_mask,
                                                                slf_attn_mask=slf_attn_mask,
                                                                dec_enc_attn_mask=dec_enc_attn_mask)        #(B,Lq,H)

        logits = torch.matmul(dec_output, self.tgt_word_emb.weight.transpose(0,1))                          #(B,Lq,H)*(H,V) -> (B,Lq,V)

        return logits


class TransformerDecoderV2(nn.Module):
    ''' A decoder model with self attention mechanism, without cross attention, with copy mechanism
        Without cross attention to context
        Input: 
            tgt_seq: Target sequence
            enc_output: Output of encoder
            config: Configuration of model, consists:
                    pad_idx: Pad index
                    num_decoder_layers: Layers of transformer decoder
                    decoder_dropout: Dropout rate of transformer decoder
    '''
    def __init__(self, config, bert_word_embedding_weights, bert_pos_embedding_weights):
        
        super(TransformerDecoderV2, self).__init__()
        
        self.vocab_size = config.vocab_size
        self.pad_idx = config.pad_idx
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        
        d_k = hidden_size // num_attention_heads
        d_v = hidden_size // num_attention_heads
        
        # -- word embedding layer
        self.tgt_word_emb = nn.Embedding(self.vocab_size, hidden_size, padding_idx=config.pad_idx)
        self.tgt_word_emb.weight = bert_word_embedding_weights
        # -- position embedding layer
        self.post_word_emb = nn.Embedding(self.vocab_size, hidden_size)  
        self.post_word_emb.weight = bert_pos_embedding_weights
        # -- Transformer decoder layer without cross attention
        self.layer_stack = nn.ModuleList([
            EncoderLayer(hidden_size, 
                        config.intermediate_size, 
                        num_attention_heads,
                        d_k, d_v, dropout=config.decoder_dropout)
            ]*config.num_decoder_layers)
        self.w_gen = nn.Linear(config.hidden_size*3, 1)        
        
    def forward(self, tgt_seq, slot_features, src_seq, enc_output):
        '''tgt_seq: Concatenated slot values  (B,J,L)
            slot_features: Hidden features of slot (B,J,H)
            src_seq: Input sequence (B,Lk)
            enc_output: Sequence of encoded context from encoder: (B,Lk,H)
        '''
        # -- Reshape input
        B,J,Lq = tgt_seq.size()
        B,Lk,H = enc_output.size()
        tgt_seq = tgt_seq.contiguous().view(B*J,-1)                         #(B,J,Lq) -> (B*J,Lq)
        slot_features = slot_features.contiguous().view(B*J,1,-1)           #(B,J,H) -> (B*J,H)
        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq, self.pad_idx)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)                                                 #(B*J,Lq,Lq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq, pad_idx=self.pad_idx)    #(B*J,Lq,Lq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)             

        # -- Forward
        max_len = tgt_seq.size()[1]
        position = torch.arange(0, max_len).long().unsqueeze(0).to(tgt_seq.device)
        dec_output = self.tgt_word_emb(tgt_seq) + self.post_word_emb(position)                              #(B*J,Lq,H)
        # -- Noting: 目标序列[SLOT]t1t2t3..[EOS]的embedding在进入decoder前 
        #            [SLOT]对应的feature要替换成为h_{[SLOT]_j} 因为h_{[SLOT]_j}
        #            和上下文通过self-attention进行了双向信息交互
        dec_input = torch.cat([slot_features, dec_output[:,1:]],dim=1)
        dec_output = dec_input
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn =  dec_layer(dec_output,
                                        non_pad_mask=non_pad_mask,
                                        slf_attn_mask=slf_attn_mask)                                        #(B*J,Lq,H)

        logits = torch.matmul(dec_output, self.tgt_word_emb.weight.transpose(0,1))                          #(B*J,Lq,H)*(H,V) -> (B*J,Lq,V)
        logits = logits.view(B,J,Lq,-1)                                                                     #(B*J,Lq,V) -> (B,J,Lq,V)
        
        dec_output = dec_output.view(B,J,Lq,-1)
        attn_e = torch.matmul(dec_output,enc_output.unsqueeze(1).transpose(-1,-2))                          #(B,J,Lq,H) * (B,1,H,Lk) -> (B,J,Lq,Lk)
        mask = src_seq.eq(self.pad_idx).view(B,1,1,Lk)                                                      #(B,1,1,Lk)
        attn_e = attn_e.masked_fill(mask, -1e9)                                                             #(B,J,Lq,Lk)
        # Copy probability
        attn_history = nn.functional.softmax(attn_e, -1)                                                    #(B,J,Lq,Lk)     

        p_vocab = nn.functional.softmax(logits, -1)                                                         #(B,J,Lq,V)

        context = torch.matmul(attn_history, enc_output.unsqueeze(1))                                       #(B,J,Lq,Lk) * (B,1,Lk,H) -> (B,J,Lq,H)
        #Pointer cofficient
        dec_input = dec_input.view(B,J,Lq,-1)
        p_gen = torch.sigmoid(self.w_gen(torch.cat([dec_input, context, dec_output], -1)))                  #(B,J,Lq,1) 
        p_gen = p_gen.squeeze(-1)

        p_context_ptr = torch.zeros_like(p_vocab).to(dec_input.device)                                      #(B,J,Lq,V)
        index = src_seq.repeat(1,J*Lq).view(B,J,Lq,-1)
        p_context_ptr.scatter_add_(-1, index, attn_history)                                                 # copy B,V
        p_final = p_gen.unsqueeze(-1) * p_vocab + (1 - p_gen.unsqueeze(-1)) * p_context_ptr                 # B,V

        return p_final


class TransformerDecoderV3(nn.Module):
    ''' A decoder model with self attention mechanism, without cross attention, with copy mechanism
        With cross attention to context
        Input: 
            tgt_seq: Target sequence
            enc_output: Output of encoder
            config: Configuration of model, consists:
                    pad_idx: Pad index
                    num_decoder_layers: Layers of transformer decoder
                    decoder_dropout: Dropout rate of transformer decoder
    '''
    def __init__(self, config, bert_word_embedding_weights, bert_pos_embedding_weights):
        
        super(TransformerDecoderV3, self).__init__()
        
        self.vocab_size = config.vocab_size
        self.pad_idx = config.pad_idx
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        
        d_k = hidden_size // num_attention_heads
        d_v = hidden_size // num_attention_heads
        
        # -- word embedding layer
        self.tgt_word_emb = nn.Embedding(self.vocab_size, hidden_size, padding_idx=config.pad_idx)
        self.tgt_word_emb.weight = bert_word_embedding_weights
        # -- position embedding layer
        self.post_word_emb = nn.Embedding(self.vocab_size, hidden_size)  
        self.post_word_emb.weight = bert_pos_embedding_weights
        # -- Transformer decoder layer without cross attention
        self.layer_stack = nn.ModuleList([
            DecoderLayer(hidden_size, 
                        config.intermediate_size, 
                        num_attention_heads,
                        d_k, d_v, dropout=config.decoder_dropout)
            ]*config.num_decoder_layers)
        self.w_gen = nn.Linear(config.hidden_size*3, 1)        
        
    def forward(self, tgt_seq, slot_features, src_seq, enc_output):
        '''tgt_seq: Concatenated slot values  (B,J,L)
            slot_features: Hidden features of slot (B,J,H)
            src_seq: Input sequence (B,Lk)
            enc_output: Sequence of encoded context from encoder: (B,Lk,H)
        '''
        # -- Reshape input
        B,J,Lq = tgt_seq.size()
        B,Lk,H = enc_output.size()
        tgt_seq = tgt_seq.contiguous().view(B*J,-1)                         #(B,J,Lq) -> (B*J,Lq)
        src_seq = src_seq.repeat(1,J).contiguous().view(B*J,-1)             #(B,Lk) -> (B*J,Lk)
        enc_output = enc_output.repeat(1,J,1).contiguous().view(B*J,Lk,-1)  #(B,Lk,H) -> (B*J,Lk,H)
        slot_features = slot_features.contiguous().view(B*J,1,-1)           #(B,J,H) -> (B*J,H)
        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq, self.pad_idx)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)                                                 #(B*J,Lq,Lq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq, pad_idx=self.pad_idx)    #(B*J,Lq,Lq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)             
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq, pad_idx=self.pad_idx)       #(B*J,Lq,Lk)

        # -- Forward
        max_len = tgt_seq.size()[1]
        position = torch.arange(0, max_len).long().unsqueeze(0).to(tgt_seq.device)
        dec_output = self.tgt_word_emb(tgt_seq) + self.post_word_emb(position)                              #(B*J,Lq,H)
        # -- Noting: 目标序列[SLOT]t1t2t3..[EOS]的embedding在进入decoder前 
        #            [SLOT]对应的feature要替换成为h_{[SLOT]_j} 因为h_{[SLOT]_j}
        #            和上下文通过self-attention进行了双向信息交互
        dec_input = torch.cat([slot_features, dec_output[:,1:]],dim=1)
        dec_output = dec_input
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, enc_output,
                                                                non_pad_mask=non_pad_mask,
                                                                slf_attn_mask=slf_attn_mask,
                                                                dec_enc_attn_mask=dec_enc_attn_mask)        #(B*J,Lq,H)

        logits = torch.matmul(dec_output, self.tgt_word_emb.weight.transpose(0,1))                          #(B*J,Lq,H)*(H,V) -> (B*J,Lq,V)
        logits = logits.view(B,J,Lq,-1)                                                                     #(B*J,Lq,V) -> (B,J,Lq,V)

        dec_output = dec_output.view(B,J,Lq,-1)
        enc_output = enc_output.view(B,J,Lk,-1)
        attn_e = torch.matmul(dec_output,enc_output.transpose(-1,-2))                                       #(B,J,Lq,H) * (B,J,H,Lk) -> (B,J,Lq,Lk)
        mask = src_seq.eq(self.pad_idx).view(B,J,1,Lk)                                                      #(B,1,1,Lk)
        attn_e = attn_e.masked_fill(mask, -1e9)                                                             #(B,J,Lq,Lk)
        # Copy probability
        attn_history = nn.functional.softmax(attn_e, -1)                                                    #(B,J,Lq,Lk)     

        p_vocab = nn.functional.softmax(logits, -1)                                                         #(B,J,Lq,V)

        context = torch.matmul(attn_history, enc_output)                                                    #(B,J,Lq,Lk) * (B,J,Lk,H) -> (B,J,Lq,H)
        #Pointer cofficient
        dec_input = dec_input.view(B,J,Lq,-1)
        p_gen = torch.sigmoid(self.w_gen(torch.cat([dec_input, context, dec_output], -1)))                  #(B,J,Lq,1) 
        p_gen = p_gen.squeeze(-1)

        p_context_ptr = torch.zeros_like(p_vocab).to(dec_input.device)                                      #(B,J,Lq,V)
        index = src_seq.repeat(1,Lq).view(B,J,Lq,-1)
        p_context_ptr.scatter_add_(-1, index, attn_history)                                                 # copy B,V
        p_final = p_gen.unsqueeze(-1) * p_vocab + (1 - p_gen.unsqueeze(-1)) * p_context_ptr                 # B,V

        return p_final