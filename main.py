import argparse
import os
import random
from utils import *
import numpy as np
import torch
import logging
from run import Run

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument('--dropout', type=float, default=0.1) 
parser.add_argument('--model_name', default='bert')
# ========================= Data Configs ==========================
parser.add_argument('--train_file',default='./data/starting_ki/train_all_tasks.csv')
parser.add_argument('--test_file',default='./data/dev_task_a_entries.csv')
parser.add_argument('--batch_size', type = int, default =32)
parser.add_argument('--num_workers', default=4, type=int, help="num_workers for dataloaders")
parser.add_argument('--labeltype', default='label_sexist')
 # ========================== BERT =============================
parser.add_argument('--bert_dir', type=str, default='bert-base-uncased')
parser.add_argument('--test_bert_dir', type=str, default='bert-base-uncased')
parser.add_argument('--bert_cache', type=str, default='./data/cache')
parser.add_argument('--bert_learning_rate', type=float, default=5e-5)
# parser.add_argument('--bert_warmup_steps', type=int, default=5000)
# parser.add_argument('--bert_max_steps', type=int, default=30000)
# parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)
# parser.add_argument("--bert_output_dim", type=float, default=768)
# parser.add_argument("--bert_hidden_size", type=float, default=768)
# ========================= Learning Configs ==========================
parser.add_argument('--max_epochs', type=int, default=30)
parser.add_argument('--print_steps', type=int, default=20, help="Number of steps to log training metrics.")
parser.add_argument('--max_steps', default=50000, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--warmup_steps', default=200, type=int, help="warm ups for parameters not in bert or vit")
parser.add_argument('--learning_rate', default=2e-4, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
# ======================== SavedModel Configs =========================
parser.add_argument('--savedmodel_path', type=str, default='./save')
parser.add_argument('--ckpt_file', type=str, default='./save/flod_/model_epoch_1_mean_f1_0.0057.bin')
parser.add_argument('--best_score', default=0.4, type=float, help='save checkpoint if mean_f1 > best_score')


args = parser.parse_args()

if __name__ == '__main__':
    setup_logging()
    setup_device(args)
    setup_seed(args)
    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("model parameters: %s", args)
    Run(args).main(args)