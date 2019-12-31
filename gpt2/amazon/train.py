import torch
from tqdm import tqdm
import torch.optim as optim
import os
import random

from transformers import *
model_class, tokenizer_class = (GPT2LMHeadModel, GPT2Tokenizer)

# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer = tokenizer_class.from_pretrained('gpt2')

model = model_class.from_pretrained('gpt2').cuda()
model.train()
print('ok')

def main():
    data_path = "/DATA/joosung/sentiment_data/Sentiment-and-Style-Transfer-master/data"
    train_amazon_neg_path = data_path + "/amazon/sentiment.train.0"
    train_amazon_neg_open = open(train_amazon_neg_path, "r")
    train_amazon_neg_dataset = train_amazon_neg_open.readlines()
    amazon_neg_dataset = train_amazon_neg_dataset
    
    neg_len = len(amazon_neg_dataset)
    train_amazon_neg_open.close()

    train_amazon_pos_path = data_path + "/amazon/sentiment.train.1"
    train_amazon_pos_open = open(train_amazon_pos_path, "r")
    train_amazon_pos_dataset = train_amazon_pos_open.readlines()
    amazon_pos_dataset = train_amazon_pos_dataset
    
    pos_len = len(amazon_pos_dataset)
    train_amazon_pos_open.close()

    epoch = 5
    epoch_len = max(pos_len,neg_len)
    stop_point = epoch_len*epoch    
    
    # Parameters:
    lr = 1e-3
    max_grad_norm = 1.0
    num_total_steps = stop_point # 1000
    num_warmup_steps = int(stop_point/10) # 100
    warmup_proportion = float(num_warmup_steps) / float(num_total_steps)  # 0.1

    lm_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)  # PyTorch scheduler    

    torch.cuda.empty_cache()
    for start in tqdm(range(stop_point)):
        """data start point"""
        neg_start = start%neg_len
        pos_start = start%pos_len

        """data setting"""
        neg_sentence = amazon_neg_dataset[neg_start].strip()
        pos_sentence = amazon_pos_dataset[pos_start].strip()                

        sentences = [neg_sentence, pos_sentence]
        """data input"""
        for i in range(2):
            # k=0: negative, k=1: positive
            sentence = sentences[i]

            sen_idx = torch.tensor(tokenizer.encode(sentence)).cuda()
            output = model(sen_idx)
            
            if len(sen_idx) == 1:
                continue
            target = sen_idx[1:]
            pred = output[0][:-1,:]            

            loss = lm_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

#             print(loss)
        if (start+1)%epoch_len == 0:
            random.shuffle(amazon_neg_dataset)
            random.shuffle(amazon_pos_dataset)
            save_model((start+1)//pos_len)        
    save_model('final') # final_model
    
    
def save_model(name):
    if not os.path.exists(str(name)+'/'):
        os.makedirs(str(name)+'/')
    model.save_pretrained('./'+str(name))
    tokenizer.save_pretrained('./'+str(name))    

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
        