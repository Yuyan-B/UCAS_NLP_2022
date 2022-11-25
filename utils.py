import os
import torch
import logging
import random
import numpy as np
from sklearn.metrics import f1_score
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn

def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()

def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def setup_logging(args):
    if not os.path.isdir(args.log_path):
        os.makedirs(args.log_path)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(args.log_path, '%s_%s.log' % (args.model_name, args.labeltype)),
        filemode='a')
    logger = logging.getLogger(__name__)
    return logger

def build_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    large_lr = ['']
    optimizer_grouped_parameters = [
        {'params': [j for i, j in model.named_parameters() if (not 'bert' in i and not any(nd in i for nd in no_decay))],
         'lr': args.learning_rate, 'weight_decay': args.weight_decay},
        {'params': [j for i, j in model.named_parameters() if (not 'bert' in i and any(nd in i for nd in no_decay))],
         'lr': args.learning_rate, 'weight_decay': 0.0},
        {'params': [j for i, j in model.named_parameters() if ('bert' in i and not any(nd in i for nd in no_decay))],
         'lr': args.bert_learning_rate, 'weight_decay': args.weight_decay},
        {'params': [j for i, j in model.named_parameters() if ('bert' in i and any(nd in i for nd in no_decay))],
         'lr': args.bert_learning_rate, 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    return optimizer, scheduler

def evaluate(predictions, labels):
    f1_macro = f1_score(labels, predictions,average='macro')#
    f1_micro = f1_score(labels, predictions,average='micro')#
    f1_weight = f1_score(labels, predictions,average='weighted')#
    eval_results = {'f1_macro':f1_macro,'f1_micro':f1_micro,'f1_weight':f1_weight}
    return eval_results

# FGM
class FGM:
    def __init__(self, model: nn.Module, eps=1.):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.eps = eps
        self.backup = {}
    # only attack word embedding
    def attack(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm and not torch.isnan(norm):
                    r_at = self.eps * param.grad / norm
                    param.data.add_(r_at)
    def restore(self, emb_name='word_embeddings'):
        for name, para in self.model.named_parameters():
            if para.requires_grad and emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]
        self.backup = {}

# PGD
class PGD:
    def __init__(self, model, eps=1., alpha=0.3):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.eps = eps
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}
    def attack(self, emb_name='word_embeddings', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)
    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}
    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.eps:
            r = self.eps * r / torch.norm(r)
        return self.emb_backup[param_name] + r
    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]