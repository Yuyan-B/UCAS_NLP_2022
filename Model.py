import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
from transformers import BertModel
from utils import *

class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.bert=BertModel.from_pretrained(args.bert_name, cache_dir=args.pretrain_cache)
        feature_kernel={1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        self.convs = CNNExtractor(feature_kernel, 768)
        mlp_input_shape = sum([feature_num for _, feature_num in feature_kernel.items()])
        if args.labeltype=='label_sexist':
            num_labels=2
        elif args.labeltype=='label_category':
            num_labels=4
        elif args.labeltype=='label_vector':
            num_labels=11
        self.mlp = MLP(mlp_input_shape, [512], num_labels, args.dropout)
        self.device=args.device

    def forward(self,batch,inference=False):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        feature=self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        output = self.convs(feature)
        output = self.mlp(output)
        logits = F.log_softmax(output,1)
        pred_label=torch.argmax(logits, dim=1)
        if inference:
            return pred_label,logits
        else:
            label = batch['label'].to(self.device)
            loss=F.nll_loss(logits,label)
            return loss,pred_label

class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        if args.labeltype=='label_sexist':
            num_labels=2
        elif args.labeltype=='label_category':
            num_labels=4
        elif args.labeltype=='label_vector':
            num_labels=11
        self.model = BertModel.from_pretrained(args.bert_name, cache_dir=args.pretrain_cache)
        self.mlp = MLP(768, [512], num_labels, args.dropout)
        self.device=args.device

    def forward(self, batch,inference=False):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        feature=self.model(input_ids, attention_mask=attention_mask).pooler_output
        output = self.mlp(feature)
        logits = F.log_softmax(output,1)
        pred_label=torch.argmax(logits, dim=1)
        if inference:
            return pred_label, logits
        else:
            label = batch['label'].to(self.device)
            loss=F.nll_loss(logits,label)
            return loss,pred_label

class RoBerta(nn.Module):
    def __init__(self, args):
        super(RoBerta, self).__init__()
        if args.labeltype=='label_sexist':
            num_labels=2
        elif args.labeltype=='label_category':
            num_labels=4
        elif args.labeltype=='label_vector':
            num_labels=11
        self.model = RobertaModel.from_pretrained(args.roberta_name, cache_dir=args.pretrain_cache)
        self.mlp = MLP(768, [512], num_labels, args.dropout)
        self.device=args.device

    def forward(self, batch,inference=False):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        feature=self.model(input_ids, attention_mask=attention_mask).pooler_output
        output = self.mlp(feature)
        logits = F.log_softmax(output,1)
        pred_label=torch.argmax(logits, dim=1)
        if inference:
            return pred_label, logits
        else:
            label = batch['label'].to(self.device)
            loss=F.nll_loss(logits,label)
            return loss,pred_label
        

class Model_MultiTask(nn.Module):
    def __init__(self, args):
        super(Model_MultiTask, self).__init__()
        if args.model_name=='roberta':
            self.model = RobertaModel.from_pretrained(args.roberta_name, cache_dir=args.pretrain_cache)
        elif args.model_name=='bert' or args.model_name=='textcnn':
            self.model = BertModel.from_pretrained(args.bert_name, cache_dir=args.pretrain_cache)
        
        feature_kernel={1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        self.convs = CNNExtractor(feature_kernel, 768)
        if args.model_name=='textcnn':
            mlp_input_shape = sum([feature_num for _, feature_num in feature_kernel.items()])
        else:
            mlp_input_shape=768
        
        self.mlpA=MLP(mlp_input_shape, [512], 2, args.dropout)
        self.mlpB=MLP(mlp_input_shape, [512], 4, args.dropout)
        self.mlpC=MLP(mlp_input_shape, [512], 11, args.dropout)
        # self.softmax=F.log_softmax(dim=1)
        self.device=args.device
        self.lambdaA=args.lambdaA
        self.lambdaB=args.lambdaB
        self.lambdaC=args.lambdaC
        self.task=args.labeltype
        self.model_name=args.model_name

    def forward(self, batch,inference=False):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        bert_out=self.model(input_ids, attention_mask=attention_mask)
        
        if self.model_name=='textcnn':
            feature = self.convs(bert_out.last_hidden_state)
        else:
            feature=bert_out.pooler_output
        
        fea_A=self.mlpA(feature)
        logits_A=F.log_softmax (fea_A,1)
        
        fea_B=self.mlpB(feature)
        logits_B=F.log_softmax (fea_B,1)
        # p_Aeq1=logits_A[:,1].reshape(-1,1).expand([logits_A.shape[0],fea_B.shape[1]])  
        # logits_B=torch.div(logits_B,p_Aeq1)   #p(b=x|A=1)=p(B=x)/p(A=1) 根据贝叶斯公式对三塔的logits做个调整
        fea_C=self.mlpC(feature)
        logits_C=F.log_softmax(fea_C,1)
        # p_B=torch.cat((logits_B[:,0].reshape(-1,1).repeat(1,2),logits_B[:,1].reshape(-1,1).repeat(1,3),logits_B[:,2].reshape(-1,1).repeat(1,4),logits_B[:,3].reshape(-1,1).repeat(1,2)),1) 
        # logits_C=torch.div(logits_C,p_B)
        if inference:
            if self.task=='label_sexist':
                pred_label=torch.argmax(logits_A, dim=1)
                logits=logits_A
            elif self.task=='label_category':
                pred_label=torch.argmax(logits_B, dim=1)
                logits=logits_B
            else:
                pred_label=torch.argmax(logits_C, dim=1)
                logits=logits_C
            return pred_label, logits
        else:
            label_A = batch['labelA'].to(self.device)
            label_B = batch['labelB'].to(self.device)
            label_C = batch['labelC'].to(self.device)
            mask_index=torch.nonzero(batch['sexist_mask']).to(self.device).squeeze(dim=1)
            lossA = F.nll_loss(logits_A, label_A)
            lossB =  F.nll_loss(torch.index_select(logits_B,0,mask_index), torch.index_select(label_B,0,mask_index))
            lossC =  F.nll_loss(torch.index_select(logits_C,0,mask_index), torch.index_select(label_C,0,mask_index))
            loss=self.lambdaA*lossA+self.lambdaB*lossB+self.lambdaC*lossC #需要对lambda进行调整
            
            if self.task=='label_sexist':
                pred_label=torch.argmax(logits_A, dim=1)
            elif self.task=='label_category':
                pred_label=torch.argmax(logits_B, dim=1)
            else:
                pred_label=torch.argmax(logits_C, dim=1)
                
            return loss,pred_label