import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.insert(0, "/DATA/joosung/fairseq_master")

class findattribute(nn.Module):
    def __init__(self, drop_rate=0, gpu = True):
        super(findattribute, self).__init__()
        self.gpu = gpu
        
        self.n_vocab = 50259
        self.emb_dim = 256
        
        """idx & length"""
        self.START_IDX = 50257
        self.PAD_IDX = 50258
        self.EOS_IDX = 50256
        
        """Discriminator(classifier)"""
        self.word_dim = 256
        self.word_emb = nn.Embedding(self.n_vocab, self.word_dim, self.PAD_IDX) # 50265x1024
        
        self.channel_out = 100
        self.conv2d_2 = nn.Conv2d(1,self.channel_out,(2,self.word_dim))
        self.conv2d_3 = nn.Conv2d(1,self.channel_out,(3,self.word_dim))
        self.conv2d_4 = nn.Conv2d(1,self.channel_out,(4,self.word_dim))
        self.conv2d_5 = nn.Conv2d(1,self.channel_out,(5,self.word_dim))
#         self.fc_drop = nn.Dropout(drop_rate)
        self.disc_fc = nn.Linear(4*self.channel_out, 2)
        
        """parameters"""                
        self.cls_params = list(self.word_emb.parameters())+list(self.conv2d_2.parameters())+list(self.conv2d_3.parameters())+list(self.conv2d_4.parameters())+\
        list(self.conv2d_5.parameters())+list(self.disc_fc.parameters())
            

    def discriminator(self, token_idx):
        """
        token_idx: (batch, seq_len)
        """
        if token_idx.shape[1] < 5:
            padding_size = 5-token_idx.shape[1]
            padding_token = []
            for k in range(token_idx.shape[0]):
                temp = []
                for i in range(padding_size):
                    temp.append(self.PAD_IDX)
                padding_token.append(temp)                
            padding_token=torch.from_numpy(np.array(padding_token))
            if self.gpu == True:
                padding_token = padding_token.cuda()
            token_idx=torch.cat([token_idx,padding_token], 1) # (batch, seq_len+padding) = (batch, 5)

        word_emb = self.word_emb(token_idx) # (batch, seq_len, word_dim)
        word_2d = word_emb.unsqueeze(1) # (batch, 1, seq_len, word_dim)

        x2 = F.relu(self.conv2d_2(word_2d)).squeeze(3) # bi-gram, (batch, channel_out, seq_len-1)
        x3 = F.relu(self.conv2d_3(word_2d)).squeeze(3) # 3-gram, (batch, channel_out, seq_len-2)
        x4 = F.relu(self.conv2d_4(word_2d)).squeeze(3) # 4-gram, (batch, channel_out, seq_len-3)
        x5 = F.relu(self.conv2d_5(word_2d)).squeeze(3) # 5-gram, (batch, channel_out, seq_len-4)

        # Max-over-time-pool
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2) # (batch, channel_out)
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2) # (batch, channel_out)
        x4 = F.max_pool1d(x4, x4.size(2)).squeeze(2) # (batch, channel_out)
        x5 = F.max_pool1d(x5, x5.size(2)).squeeze(2) # (batch, channel_out)
        x = torch.cat([x2, x3, x4, x5], dim=1) # (batch, channel_out*4)

        y = self.disc_fc(x) # (batch, 2)

        if self.gpu == True:
            return y.cuda()
        else:
            return y
        
    def gen_discriminator(self, gen_out):
        """
        gen_out: (gen_len+2, batch, n_vocab)
        """
        gen_emb = gen_out[1:-1,:,:] # (gen_len, batch, n_vocab)
        gen_emb = torch.bmm(gen_emb, self.word_emb.weight.repeat(gen_emb.shape[0],1,1))
        # (gen_len, batch, emb_dim) = (gen_len, batch, n_vocab) x (gen_len, n_vocab, emb_dim)
        gen_emb = gen_emb.transpose(0, 1) # (batch, gen_len, word_dim)
        
        if gen_emb.shape[1] < 5:
            padding_size = 5-gen_emb.shape[1]
            padding_token = []
            for k in range(gen_emb.shape[0]):
                temp = []
                for i in range(padding_size):
                    temp.append(self.PAD_IDX)
                padding_token.append(temp)                
            padding_token=torch.from_numpy(np.array(padding_token)) # (batch, padding_len)
            if self.gpu == True:
                padding_token = padding_token.cuda()
            padding_emb = self.word_emb(padding_token) # (batch, padding_len, emb_dim)
            gen_emb = torch.cat([gen_emb, padding_emb], 1) # (batch, 5, emb_dim)   
            
        word_2d = gen_emb.unsqueeze(1) # (batch, 1, seq_len, word_dim)

        x2 = F.relu(self.conv2d_2(word_2d)).squeeze(3) # bi-gram, (batch, channel_out, seq_len-1)
        x3 = F.relu(self.conv2d_3(word_2d)).squeeze(3) # 3-gram, (batch, channel_out, seq_len-2)
        x4 = F.relu(self.conv2d_4(word_2d)).squeeze(3) # 4-gram, (batch, channel_out, seq_len-3)
        x5 = F.relu(self.conv2d_5(word_2d)).squeeze(3) # 5-gram, (batch, channel_out, seq_len-4)

        # Max-over-time-pool
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2) # (batch, channel_out)
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2) # (batch, channel_out)
        x4 = F.max_pool1d(x4, x4.size(2)).squeeze(2) # (batch, channel_out)
        x5 = F.max_pool1d(x5, x5.size(2)).squeeze(2) # (batch, channel_out)
        x = torch.cat([x2, x3, x4, x5], dim=1) # (batch, channel_out*4)

        y = self.disc_fc(x) # (batch, 2)

        if self.gpu == True:
            return y.cuda()
        else:
            return y
        
    def att_prob(self, token_idx, sentiment):
        """
        token_idx: (batch, seq_len)
        """
#         if token_idx.size(1) < 5:
#             padding_size = 5-token_idx.size(1)
#             padding_token = []
#             for k in range(token_idx.size(0)):
#                 temp = []
#                 for i in range(padding_size):
#                     temp.append(self.PAD_IDX)
#                 padding_token.append(temp)                
#             padding_token=torch.from_numpy(np.array(padding_token))
#             if self.gpu == True:
#                 padding_token = padding_token.cuda()
#             token_idx=torch.cat([token_idx,padding_token], 1) # (batch, seq_len+padding) = (batch, 5)
        token_list = token_idx.squeeze(0).cpu().tolist() # list
        min_prob = 1
        for i in range(len(token_list)):
            del_list = token_list[:i] + token_list[i+1:]
            del_tensor = torch.from_numpy(np.asarray(del_list)).unsqueeze(0).cuda()
            del_prob=F.softmax(self.discriminator(del_tensor),1).squeeze(0)[sentiment].cpu().detach().numpy().item()
            
            if del_prob <= min_prob:                
                max_ind = i
                min_prob = del_prob
                
        final_list = token_list[:max_ind] + token_list[max_ind+1:]
        del_idx = torch.from_numpy(np.asarray(final_list)).unsqueeze(0).cuda()
        return del_idx    
        
    def cls_loss(self, targets, cls_out):
        """
        targets: (batch, 2) / attributes [0,1] or [1,0]
        cls_out: (batch, 2) (logits)
        """
        
        final_targets = targets.argmax(1) # (batch)
        cls_loss = F.cross_entropy(cls_out, final_targets)
        
        if self.gpu == True:       
            return cls_loss.cuda()
        else:
            return cls_loss
    






