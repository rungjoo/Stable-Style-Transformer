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
from dis_model import *
dismodel = findattribute(drop_rate = 0.4).cuda()
dismodel.load_state_dict(torch.load('../visual_v1_0/models/cls_model_final'))
dismodel.train()

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
    cls_initial_lr = 0.001
    cls_trainer = optim.Adamax(dismodel.cls_params, lr=cls_initial_lr) # initial 0.001
    max_grad_norm = 25
    batch = 1
    epoch = 6
    stop_point = pos_len*epoch
    
    pre_epoch = 0
    for start in tqdm(range(0, stop_point)):            
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

        """data input"""
        for i in range(2):
            # k=0: negative, k=1: positive
            sentence = sentences[i]
            attribute = attributes[i] # for generate

            token_idx = torch.tensor(gpt_tokenizer.encode(sentence)).unsqueeze(0).cuda()
            
            dis_out = dismodel.discriminator(token_idx)

            """calculation loss & traning"""
            # training using discriminator loss
            cls_loss = dismodel.cls_loss(attribute, dis_out)
            summary.add_scalar('discriminator loss', cls_loss.item(), start)

            cls_trainer.zero_grad()
            cls_loss.backward() # retain_graph=True
            grad_norm = torch.nn.utils.clip_grad_norm_(dismodel.cls_params, max_grad_norm)            
            cls_trainer.step()
        
        """savining point"""
        if (start+1)%pos_len == 0:
            random.shuffle(amazon_neg_dataset)
            random.shuffle(amazon_pos_dataset)
            save_model((start+1)//pos_len)        
    save_model('final') # final_model    

    
def save_model(iter):
    if not os.path.exists('models/'):
        os.makedirs('models/')
    torch.save(dismodel.state_dict(), 'models/cls_model_{}'.format(iter))  
    

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
    
