from sklearn.model_selection import  train_test_split,StratifiedShuffleSplit
from transformers import BertTokenizer,RobertaTokenizer
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset

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

        text = item['text']
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

        text = item['text']
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