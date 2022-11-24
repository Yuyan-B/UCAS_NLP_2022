from sklearn.model_selection import  train_test_split,StratifiedShuffleSplit
from transformers import BertTokenizer,RobertaTokenizer
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import re
import nltk

def cleanText(text):
    def add_space(matched):
        s = matched.group()
        return ' '+ s[0] + ' ' + s[-1]
    
    con_cleaned = re.sub(r'[^a-zA-Z0-9_\-\.,;:!?/\']', " ", text)#替换函数
    con_cleaned = re.sub(r'[\.,;:!?/]+[a-zA-Z]', add_space, con_cleaned)#这一行将前面符号后面文字变成了 空格符号空格
    #con_cleaned = re.sub(r'[^a-zA-Z\.,;:!?/\']', add_space, con_cleaned)
    
    try:
        #print(nltk.word_tokenize("fdsf dczxf wen wafzd da."))分词不好用？？？
        wordtoken = nltk.word_tokenize(con_cleaned)#这一行发生了异常
        #这里记录一下，因为需要nltk.download('punkt')这一行，才可以使用nltk的分词功能，试了一天终于发现原来是这里出现了问题，jupyter里面试出来的
    except:
        print(con_cleaned)
        print(text)
        exit()#这里直接退出了
    content_tackled = ' '.join(wordtoken)

    def add_space_pre(matched):
        '''
        If word like "china." occured, split "china" and ".". 
        '''
        s = matched.group()
        return s[0] + ' ' + s[-1]
        
    content_tackled = re.sub(r'[a-zA-Z][\.,;:!?/]+', add_space_pre, content_tackled)
    
    return content_tackled

class TrainSexismDataset(Dataset):
    def __init__(self, args,data,le):
        self.data=data
        #将数据id转为整数
        self.data['id']=self.data['rewire_id'].apply(lambda x:int(x[19:]))
        #对不同的TASK的标签编码
        self.data['label']=le.transform(self.data[args.labeltype])
        if "roberta" in args.bert_dir:
            self.tokenizer = RobertaTokenizer.from_pretrained(args.bert_dir,cache_dir=args.bert_cache)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir,cache_dir=args.bert_cache)
        
    def __len__(self):
        return self.data.shape[0]
     
    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        text = cleanText(item['text'])
        tokens = self.tokenizer(text, max_length=250, padding='max_length', truncation=True)
        text_inputid = torch.tensor(tokens['input_ids'])
        text_mask = torch.tensor(tokens['attention_mask'])
        label = torch.tensor(item['label'])
        id=torch.tensor(item['id'])

        return {
            'id':id,
            'label': label,
            'input_ids': text_inputid,
            'attention_mask': text_mask,
        } 

class TestSexismDataset(Dataset):
    def __init__(self, args,data):
        self.data=data
        #将数据id转为整数
        self.data['id']=self.data['rewire_id'].apply(lambda x:int(x[19:]))
        
        if "roberta" in args.bert_dir:
            self.tokenizer = RobertaTokenizer.from_pretrained(args.bert_dir,cache_dir=args.bert_cache)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir,cache_dir=args.bert_cache)
        
    def __len__(self):
        return self.data.shape[0]
     
    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        text = cleanText(item['text'])
        tokens = self.tokenizer(text, max_length=250, padding='max_length', truncation=True)
        text_inputid = torch.tensor(tokens['input_ids'])
        text_mask = torch.tensor(tokens['attention_mask'])

        id=torch.tensor(item['id'])

        return {
            'id':id,
            'input_ids': text_inputid,
            'attention_mask': text_mask,
        }  

def create_train_dataloaders(args,le):
    raw_data=pd.read_csv(args.train_file)
    if args.labeltype=='label_sexist':
        raw_data=raw_data[raw_data['raw_data']=='sexist']
    train_set,val_set =  train_test_split(raw_data, test_size=0.2, stratify=raw_data[args.labeltype])
    dtrain=TrainSexismDataset(args,train_set,le)
    dval=TrainSexismDataset(args,val_set,le)
    train_dataloader = DataLoader(dtrain, batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
    val_dataloader = DataLoader(dval, batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
    return train_dataloader, val_dataloader

def create_test_dataloaders(args):
    raw_data=pd.read_csv(args.test_file)
    dtest=TestSexismDataset(args,raw_data)
    test_dataloader = DataLoader(dtest, batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
    return test_dataloader