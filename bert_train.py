from tqdm import tqdm
import os
import random
import torch

from bert_model import *
my_model = mymodel().cuda()
my_model.train()
print('ok')

## finetune bert_model
def main():
    f_label = open('main_action.txt')
    labels = f_label.readlines()
    gts = [x.strip() for x in labels]
        
    text_data_num = 300000
    f = open('corpus_300000.txt')
    texts = f.readlines()    

    # Parameters:
    lr = 2e-5 # 1e-3
    max_grad_norm = 10
#     num_training_steps = 10*len(texts)
#     num_warmup_steps = 100
#     warmup_proportion = float(num_warmup_steps) / float(num_training_steps)  # 0.1

    ### In Transformers, optimizer and schedules are splitted and instantiated like this:
    optimizer = torch.optim.AdamW(my_model.parameters(), lr=lr, eps=1e-06, weight_decay=0.01)
#     optimizer = AdamW(my_model.parameters(), lr=lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
#     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler
    
    
    for epoch in tqdm(range(3)):
        for i in range(len(texts)):
            label = texts[i].split('\t')[0]    
            sen = texts[i].split('\t')[1].strip()

            label_idx = gts.index(label)
            torch_label = torch.from_numpy(np.asarray([label_idx])).cuda() 

            bert_out = my_model.bert_layer(sen) # (batch, seq_len, emb_dim)    

            fc_out = my_model.fc_layer(bert_out)
            cls_out = fc_out[:,0:1,:].squeeze(1)

            loss = my_model.cls_loss(torch_label, cls_out)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(my_model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
#             scheduler.step()
            optimizer.zero_grad()    
        
        """savining point"""
        if (epoch+1)%1 == 0:
            save_model((epoch+1), text_data_num)            
            random.shuffle(texts)
    save_model('final', text_data_num) # final_model            
        
        
def save_model(iter, text_data_num):
    save_path = 'bert_models/'+str(text_data_num)+'_'+str(iter)+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(my_model.state_dict(), save_path+'bert_model_{}'.format(iter))
    
    ### Now let's save our model and tokenizer to a directory
    my_model.bert_model.save_pretrained(save_path)
    my_model.tokenizer.save_pretrained(save_path)
    

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()        