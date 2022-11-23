import logging
import random
import numpy as np
from sklearn.metrics import f1_score
from transformers import AdamW, get_linear_schedule_with_warmup

def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()

def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def setup_logging(args):
    logging.basicConfig(
        filename='log/%s_%s.log' % (args.model_name, args.labeltype),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO)
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