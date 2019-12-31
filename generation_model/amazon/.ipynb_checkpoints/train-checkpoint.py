import logging
logger = logging.getLogger()
logger.setLevel("ERROR")

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
import os
import random

from transformers import *
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
from tqdm import tqdm
import json


## 초기화
from gen_model import *
genmodel = styletransfer().cuda()
genmodel.load_state_dict(torch.load('../ST_v2.0/models/gen_model_5'))
genmodel.train()

import sys
sys.path.insert(0, "/DATA/joosung/controllable_english/amazon/classifier/")
from dis_model import *
dismodel = findattribute().cuda()
dismodel_name='cls_model_6'
dismodel.load_state_dict(torch.load('../classifier/models/{}'.format(dismodel_name)))
dismodel.eval()

import torch.optim as optim

from tensorboardX import SummaryWriter
summary = SummaryWriter(logdir='./logs')

def main():    
    f = open('amazon_vocab.json')
    token2num = json.load(f)

    num2token = {}
    for key, value in token2num.items():
        num2token[value] = key
    f.close()

    data_path = "/DATA/joosung/sentiment_data/Sentiment-and-Style-Transfer-master/data"
    train_amazon_neg_path = data_path + "/amazon/sentiment.train.0"
    train_amazon_neg_open = open(train_amazon_neg_path, "r")
    train_amazon_neg_dataset = train_amazon_neg_open.readlines()
    dev_amazon_neg_path = data_path + "/amazon/sentiment.dev.0"
    dev_amazon_neg_open = open(dev_amazon_neg_path, "r")
    dev_amazon_neg_dataset = dev_amazon_neg_open.readlines()
    amazon_neg_dataset = train_amazon_neg_dataset+dev_amazon_neg_dataset
    
    neg_len = len(amazon_neg_dataset)
    train_amazon_neg_open.close()
    dev_amazon_neg_open.close()

    train_amazon_pos_path = data_path + "/amazon/sentiment.train.1"
    train_amazon_pos_open = open(train_amazon_pos_path, "r")
    train_amazon_pos_dataset = train_amazon_pos_open.readlines()
    dev_amazon_pos_path = data_path + "/amazon/sentiment.dev.1"
    dev_amazon_pos_open = open(dev_amazon_pos_path, "r")
    dev_amazon_pos_dataset = dev_amazon_pos_open.readlines()
    amazon_pos_dataset = train_amazon_pos_dataset+dev_amazon_pos_dataset
    
    pos_len = len(amazon_pos_dataset)
    train_amazon_pos_open.close()
    dev_amazon_pos_open.close()      

    """training parameter"""
    aed_initial_lr = 0.00001
    gen_initial_lr = 0.001
    aed_trainer = optim.Adamax(genmodel.aed_params, lr=aed_initial_lr) # initial 0.0005
    gen_trainer = optim.Adamax(genmodel.aed_params, lr=gen_initial_lr) # initial 0.0001
    max_grad_norm = 10
    batch = 1
    epoch = 6
    epoch_len = max(pos_len,neg_len)
    stop_point = epoch_len*epoch
    
    pre_epoch = 0
    for start in tqdm(range(0, stop_point)):
        ## learing rate decay
        now_epoch = (start+1)//pos_len
            
        """data start point"""
        neg_start = start%neg_len
        pos_start = start%pos_len

        """data setting"""
        neg_sentence = amazon_neg_dataset[neg_start].strip()
        pos_sentence = amazon_pos_dataset[pos_start].strip()                

        neg_labels = [] # negative labels
        neg_labels.append([1,0])
        neg_attribute = torch.from_numpy(np.asarray(neg_labels)).type(torch.FloatTensor).cuda()

        pos_labels = [] # positive labels
        pos_labels.append([0,1])
        pos_attribute = torch.from_numpy(np.asarray(pos_labels)).type(torch.FloatTensor).cuda()

        sentences = [neg_sentence, pos_sentence]
        attributes = [neg_attribute, pos_attribute]
        sentiments = [0, 1]

        """data input"""
        for i in range(2):
            # k=0: negative, k=1: positive
            sentence = sentences[i]
            attribute = attributes[i] # for decoder
            fake_attribute = attributes[abs(1-i)] # for generate
#             sentiment = sentiments[i] # for delete

            token_idx = torch.tensor(gpt_tokenizer.encode(sentence)).unsqueeze(0).cuda()

            # delete model
            max_len = int(token_idx.shape[1]/2)
            dis_out = dismodel.discriminator(token_idx)    
            sentiment = dis_out.argmax(1).cpu().item() ## 변경점 for delete
            
            del_idx = token_idx
            for k in range(max_len):
                del_idx = dismodel.att_prob(del_idx, sentiment)                
                dis_out = dismodel.discriminator(del_idx)    
                sent_porb = F.softmax(dis_out, 1).squeeze(0)[sentiment].cpu().detach().numpy().item()
                if sent_porb < 0.7:
                    break       
                    
            """auto-encoder loss & traning"""
            # training using discriminator loss
            enc_out = genmodel.encoder(del_idx)
            dec_out, vocab_out = genmodel.decoder(enc_out, token_idx, attribute)

            ## calculation loss
            recon_loss = genmodel.recon_loss(token_idx, vocab_out)
            summary.add_scalar('reconstruction loss', recon_loss.item(), start)
            
            aed_trainer.zero_grad()
            recon_loss.backward(retain_graph=True) # retain_graph=True
            grad_norm = torch.nn.utils.clip_grad_norm_(genmodel.aed_params, max_grad_norm)            
            aed_trainer.step()
            
            """decoder classification loss & training"""            
            ## calculation loss
            gen_cls_out = dismodel.gen_discriminator(vocab_out)

            ## calculation loss
            gen_cls_loss = genmodel.cls_loss(attribute, gen_cls_out)
            summary.add_scalar('generated sentence loss', gen_cls_loss.item(), start)

            gen_trainer.zero_grad()
            gen_cls_loss.backward() # retain_graph=True
            grad_norm = torch.nn.utils.clip_grad_norm_(genmodel.aed_params, max_grad_norm)
            gen_trainer.step()
            
        
        """savining point"""
        if (start+1)%epoch_len == 0:
            random.shuffle(amazon_neg_dataset)
            random.shuffle(amazon_pos_dataset)
            save_model((start+1)//pos_len)        
    save_model('final') # final_model    

    
def save_model(iter):
    if not os.path.exists('models/'):
        os.makedirs('models/')
    torch.save(genmodel.state_dict(), 'models/gen_model_{}'.format(iter))  
    

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
    
