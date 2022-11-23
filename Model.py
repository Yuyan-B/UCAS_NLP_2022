import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaForSequenceClassification,BertForSequenceClassification

class BertModel(nn.Module):
    def __init__(self, args):
        super(BertModel, self).__init__()
        if args.labeltype=='label_sexist':
            num_labels=2
        elif args.labeltype=='label_category':
            num_labels=4
        elif args.labeltype=='label_vector':
            num_labels=11
        self.bert = BertForSequenceClassification.from_pretrained(args.bert_dir, num_labels = num_labels)
        self.device=args.device

    def forward(self, batch,inference=False):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        if inference:
            logits= outputs[0]
            pred_label=torch.argmax(logits, dim=1)
            return pred_label
        else:
            loss, logits = outputs[:2]
            pred_label=torch.argmax(logits, dim=1)
            return loss,pred_label
        

    