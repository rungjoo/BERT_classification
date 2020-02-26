import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math

from transformers import *

class mymodel(nn.Module):
    def __init__(self, drop_rate=0, gpu = True):
        super(mymodel, self).__init__()
        model_class, tokenizer_class, pretrained_weights  = (BertModel, BertTokenizer, 'bert-base-uncased')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.bert_model = model_class.from_pretrained(pretrained_weights).cuda()
#         model.train()

        self.gpu = gpu
        
        self.emb_dim = 768
        self.class_num = 193
        
        self.matrix = nn.Linear(self.emb_dim, self.class_num)
        
        self.model_params = list(self.matrix.parameters())
        

    """Modeling"""
    def bert_layer(self, sen):
        sen_idx = torch.tensor([self.tokenizer.encode(sen)]).cuda()
        model_out = self.bert_model(sen_idx)[0] # (batch, seq_len, emb_dim)    
        
        return model_out
        
    
    def fc_layer(self, bert_output):
        """
        bert_output: (batch, seq_len, emb_dim)
        """
        fc_out = self.matrix(bert_output) # (batch, seq_len, class_num)
        
        return fc_out
        
    def cls_loss(self, gt_idx, cls_out):
        """
        gt_idx: (batch)
        cls_out: (batch, class_num) (logits)
        """        
        cls_loss = F.cross_entropy(cls_out, gt_idx)
        
        if self.gpu == True:       
            return cls_loss.cuda()
        else:
            return cls_loss
        