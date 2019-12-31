import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
from transformers import *

sys.path.insert(0, "/DATA/joosung/fairseq_master")

import json
f = open('gpt_yelp_vocab.json')
token2num = json.load(f)

num2token = {}
for key, value in token2num.items():
    num2token[value] = key

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class styletransfer(nn.Module):
    def __init__(self, drop_rate=0, gpu = True):
        super(styletransfer, self).__init__()
        self.gpu = gpu
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        """hyper parameters"""
        self.n_vocab = 50259
        self.emb_dim = 256
        self.nhead = 4
        self.num_layers = 3
        
        """idx & length"""
        self.START_IDX = 50257
        self.PAD_IDX = 50258
        self.EOS_IDX = 50256
        self.MAX_SENT_LEN = 10
        
        """attribute matrix"""
        ## one_hot encoding
        self.att_num = 2
        self.matrix_A = nn.Linear(self.att_num, self.emb_dim)
        
        """word embedding"""
        self.emb_matrix = nn.Embedding(self.n_vocab, self.emb_dim, self.PAD_IDX) # 50259x1024
        
        """Position embedding"""
        self.pos_encoder = PositionalEncoding(self.emb_dim)
        
        """Encoder"""
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)       
        
        """Decoder"""                
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.emb_dim, nhead=self.nhead)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.num_layers)
        self.matrix_D = nn.Linear(self.emb_dim, self.n_vocab) # emb_dim -> n_vocab
        
        """parameters"""        
        self.enc_params = list(self.encoder_layer.parameters())+list(self.transformer_encoder.parameters())
        self.dec_params = list(self.decoder_layer.parameters())+list(self.transformer_decoder.parameters())+list(self.matrix_D.parameters())
        self.aed_params = list(self.emb_matrix.parameters())+self.enc_params+self.dec_params

    """Modeling"""
    def encoder(self, enc_input):
        """
        enc_input: (batch, enc_len)
        """
        word_emb = self.emb_matrix(enc_input) # (batch, enc_len, emb_dim)
        word_emb = word_emb.transpose(0, 1) # (enc_len, batch, emb_dim)
        word_pos = self.pos_encoder(word_emb) # (enc_len, batch, emb_dim)
        out_enc = self.transformer_encoder(word_pos) # (enc_len, batch, emb_dim)
        
        return out_enc
        
    def decoder(self, enc_out, dec_input, attribute):
        """
        enc_out: (enc_len, batch, emb_dim)
        dec_input: (batch, dec_len)
        attributes: (batch, 2)
        """
        att_emb = self.matrix_A(attribute).unsqueeze(0) # (1. batch, emb_dim)
        
        word_emb = self.emb_matrix(dec_input) # (batch, dec_len, emb_dim)
        word_emb = word_emb.transpose(0, 1) # (dec_len, batch, emb_dim)
        word_pos = self.pos_encoder(word_emb) # (dec_len, batch, emb_dim)    
        
        start_token = self.emb_matrix(torch.tensor(self.START_IDX).cuda()) # (emb_dim)
        start_token = start_token.repeat(1, dec_input.shape[0], 1) # (1, batch, emb_dim)        
        style_dec_input = torch.cat([att_emb, start_token, word_pos], 0) # (dec_len+2, batch, emb_dim) w/ [att], [start]
        
        tgt_mask = self.generate_square_subsequent_mask(style_dec_input.shape[0]).cuda() # (dec_len+2, dec_len+2)

        dec_out = self.transformer_decoder(style_dec_input, enc_out, tgt_mask=tgt_mask) # (dec_len+2, batch, emb_dim)
        vocab_out = self.matrix_D(dec_out) # (dec_len+2, batch, n_vocab)
        return dec_out, vocab_out
    
    def generator(self, enc_out, gen_len, attribute):
        """
        enc_out: (enc_len, batch, emb_dim)
        attributes: (batch, 2)
        gen_len: len(dec_in)+1
        """
        # initialization because there are no first token
        batch = enc_out.shape[1]
        att_emb = self.matrix_A(attribute).unsqueeze(0) # (1. batch, emb_dim)
        start_token = self.emb_matrix(torch.tensor(self.START_IDX).cuda()) # (emb_dim)
        start_token = start_token.repeat(1, batch, 1) # (1, batch, emb_dim)        
        gen_input = torch.cat([att_emb, start_token], 0) # (2, batch, emb_dim) w/ [att], [start]
        
        for i in range(gen_len):
            tgt_mask = self.generate_square_subsequent_mask(gen_input.shape[0]).cuda() # (pre_gen_len, pre_gen_len)
            dec_out = self.transformer_decoder(gen_input, enc_out, tgt_mask=tgt_mask) # (pre_gen_len, batch, emb_dim)
            vocab_out = self.matrix_D(dec_out) # (pre_gen_len, batch, n_vocab)
            
            vocab_idx = vocab_out.argmax(2) # (pre_gen_len, batch)
            vocab_idx = vocab_idx.transpose(0, 1) # (batch, pre_gen_len)
            
            new_word_emb = self.emb_matrix(vocab_idx) # (batch, pre_gen_len, emb_dim)
            new_word_emb = new_word_emb.transpose(0, 1) # (pre_gen_len, batch, emb_dim)
#             gen_emb = torch.bmm(vocab_out, self.emb_matrix.weight.repeat(vocab_out.shape[0],1,1))
            
#             word_pos = self.pos_encoder(word_emb) # (enc_len, batch, emb_dim)
            gen_input = torch.cat([gen_input, new_word_emb[-1:,:,:]]) # (pre_gen_len+1, batch, word_dim), pre_gen_len+=1        
        
        return vocab_out # (gen_len+2, batch, n_vocab)

    def generate_square_subsequent_mask(self,sz): # len(sz)
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    """calculation loss"""
    def recon_loss(self, dec_input, vocab_out):
        """
        dec_input: (batch, dec_len)
        vocab_out: (dec_len+2, batch, n_vocab) with [att], [start]
        """
        end_token = torch.tensor(self.EOS_IDX).cuda() # (1)
        end_token = end_token.repeat(dec_input.shape[0], 1) # (batch, 1)
        target_tokens = torch.cat([dec_input, end_token], 1) # (batch, dec_len+1) w/ [EOS]
        
        pred_out = vocab_out[1:,:,:] # (dec_len+1, batch, n_vocab)
        pred_out = pred_out.permute(1,0,2) # (batch, dec_len+1, n_vocab)
                
        target_tokens = target_tokens.contiguous() # (batch, dec_len+1)
        pred_out = pred_out.contiguous() # (batch, dec_len+1, n_vocab)
    
        target_tokens = target_tokens.view(-1) # (batch*(dec_len+1))
        pred_out = pred_out.view(-1, pred_out.shape[2]) # (batch*(seq_len+1), n_vocab)
        
        recon_loss = F.cross_entropy(pred_out, target_tokens)                
        
        return recon_loss
    
    def cls_loss(self, attributes, cls_out):
        """
        attributes: [0,1] or [1,0]
        cls_out: (batch, 2) (logits)
        """        
        targets = attributes.argmax(1) # (batch)
        cls_loss = F.cross_entropy(cls_out, targets)
        
        if self.gpu == True:       
            return cls_loss.cuda()
        else:
            return cls_loss
        
    """inferenece"""
    def dec2sen(self, vocab_out):
        """
        vocab_out: (dec_len+2, batch, n_vocab) with att, start
        """
        pred_out = vocab_out[1:,:,:] # (dec_len+1, batch, n_vocab) with [END]
        pred_idx = torch.argmax(pred_out, 2) # (dec_len+1, batch)
        pred_idx = pred_idx.squeeze(1) # (dec_len+1) because of batch=1
        
        token_list = []
        dec_sen =''
        for i in range(len(pred_idx)):
            token = num2token[pred_idx[i].cpu().numpy().item()]
            token_list.append(token)
            
            if 'Ġ' in token:
                token = token.strip('Ġ')
                dec_sen += ' '
                dec_sen += token
            else:
                dec_sen += token
        dec_sen = dec_sen.strip()
            
        
        return token_list, dec_sen
    
    def generated_sentence(self, enc_out, attribute, ori_length):
        """
        enc_out: (enc_len, batch, emb_dim)
        dec_input: (batch, dec_len)
        attributes: (batch, 2)
        """
        batch = enc_out.shape[1]
#         max_len = enc_out.shape[0]+3
        max_len = ori_length+5
        
        # initialization because there are no first token
        att_emb = self.matrix_A(attribute).unsqueeze(0) # (1. batch, emb_dim)
        start_token = self.emb_matrix(torch.tensor(self.START_IDX).cuda()) # (emb_dim)
        start_token = start_token.repeat(1, batch, 1) # (1, batch, emb_dim)        
        gen_input = torch.cat([att_emb, start_token], 0) # (2, batch, emb_dim) w/ [att], [start]

        tgt_mask = self.generate_square_subsequent_mask(gen_input.shape[0]).cuda() # (2, 2)        
        
        dec_out = self.transformer_decoder(gen_input, enc_out, tgt_mask=tgt_mask) # (2, batch, emb_dim)
        vocab_out = self.matrix_D(dec_out) # (2, batch, n_vocab)
        _, dec_sen = self.dec2sen(vocab_out)
        
        gen_vocab_out = []
        for i in range(max_len):            
            token_idx = torch.tensor(self.gpt_tokenizer.encode(dec_sen)).unsqueeze(0).cuda() # (batch, gen_len)
            if self.EOS_IDX in token_idx:
                break
                
            dec_out, vocab_out = self.decoder(enc_out, token_idx, attribute) # (dec_len+2, batch, emb_dim), (dec_len+2, batch, n_vocab)
            dec_tokens, dec_sen = self.dec2sen(vocab_out)           
            
        return dec_sen

            
            
            
            
            
            
            
            
            
            
            
            
    






