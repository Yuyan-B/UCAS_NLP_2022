import torch
import numpy as np
import pandas as pd
import logging,time
from utils import *
from datamanager import *
from sklearn.preprocessing import LabelEncoder
from Model import *
import os
from tqdm import tqdm

class Run():
    def __init__(self,args):
        self.le=LabelEncoder()
        raw_data=pd.read_csv(args.train_file)
        self.le.fit(raw_data[args.labeltype])
        self.args=args

    def train_and_validate(self, fold_id=-1):
        def train_step():
            model.train()
            optimizer.zero_grad()
            loss, pred_label = model(batch)
            loss.backward()
            if fgm is not None:
                loss_adv, _ = model(batch)
                loss_adv.backward()
                fgm.restore()
            elif pgd is not None:
                pgd.backup_grad()
                for _t in range(pgd_k):
                    pgd.attack(is_first_attack=(_t == 0))
                    if _t != pgd_k - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    loss_adv, _ = model(batch)
                    loss_adv.backward()
                pgd.restore()
            optimizer.step()
            model.zero_grad()
            scheduler.step()
            losses.append(loss.item())

        # 1. load data
        train_dataloader, val_dataloader = create_train_dataloaders(self.args, self.le, fold_id=fold_id)
        # 2. build model and optimizers
        if self.args.model_name=='bert':
            model = BertModel(self.args)
        elif self.args.model_name=='roberta':
            model = RoBertaModel(self.args)
        optimizer, scheduler = build_optimizer(self.args, model)
        model.to(self.args.device)
        fgm, pgd = None, None
        if self.args.attack == 'fgm':
            fgm = FGM(model=model)
            print('fgming')
        elif self.args.attack == 'pgd':
            pgd = PGD(model=model)
            pgd_k = 3
            print('pgding')
        # 3. training
        step = 0
        best_score = self.args.best_score
        update=0
        if self.args.oversampling:
            it = iter(train_dataloader[0])
        elif self.args.undersampling:
            it = iter(train_dataloader[1])
        for epoch in range(self.args.max_epochs):
            print('----epoch %d----' % epoch)
            losses = []
            if update>=self.args.early_stop:
                break
            if self.args.oversampling:
                for i, neg_batch in enumerate(tqdm(train_dataloader[1])):
                    # 超采样pos_batch
                    try:
                        pos_batch = next(it)
                    except:
                        it = iter(train_dataloader[0])
                        pos_batch = next(it)
                    # 合并pos_batch和neg_batch
                    batch = dict()
                    for k in pos_batch.keys():
                        batch[k] = torch.cat([pos_batch[k], neg_batch[k]])
                    train_step()
                    step += 1
            elif self.args.undersampling:
                for i, pos_batch in enumerate(tqdm(train_dataloader[0])):
                    # 欠采样neg_batch
                    try:
                        neg_batch = next(it)
                    except:
                        it = iter(train_dataloader[1])
                        neg_batch = next(it)
                    # 合并pos_batch和neg_batch
                    batch = dict()
                    for k in pos_batch.keys():
                        batch[k] = torch.cat([pos_batch[k], neg_batch[k]])
                    train_step()
                    step += 1
            else:
                for i, batch in enumerate(tqdm(train_dataloader)):
                    train_step()
                    step += 1
            # 4. validation
            if not self.args.full_train:
                loss, results = self.validate(model, val_dataloader)
                results = {k: round(v, 4) for k, v in results.items()}
                logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
            # 5. save checkpoint
            if self.args.full_train:
                epoch_loss = np.mean(losses)
                logging.info(f"Epoch {epoch} step {step}: loss {epoch_loss:.3f}")
                torch.save({'model_state_dict': model.state_dict()},
                            f'{self.args.savedmodel_path}/model_{self.args.model_name}_epoch_{epoch}_loss_{epoch_loss:.3f}.bin')
            else:
                mean_f1 = results['f1_macro']
                update += 1
                if mean_f1 > best_score:
                    best_score = mean_f1
                    logging.info(f'best_score: {best_score}')
                    update = 0
                    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'mean_f1': mean_f1},
                            f'{self.args.savedmodel_path}/model_{self.args.model_name}_epoch_{epoch}_mean_f1_{mean_f1}.bin')
        return best_score
    
    def validate(self,model, val_dataloader):
        model.eval()
        predictions = []
        labels = []
        losses = []
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                loss,pred_label = model(batch)
                loss = loss.mean()
                predictions.extend(pred_label.cpu().numpy())
                labels.extend(batch['label'].cpu().numpy())
                losses.append(loss.cpu().numpy())
        loss = np.mean(losses)
        results = evaluate(predictions, labels)
        model.train()
        return loss, results

    def inference(self):
        dataloader = create_test_dataloaders(self.args)
        # 2. load model
        if self.args.model_name=='bert':
            model = BertModel(self.args)
        elif self.args.model_name=='roberta':
            model = RoBertaModel(self.args)
        # print(model.state_dict())
        checkpoint = torch.load(self.args.ckpt_file)
        # print(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        # print(model.state_dict())
        model.cuda()
        model.eval()
        # 3. inference
        predictions = []
        id=[]
        with torch.no_grad():
            for batch in dataloader:
                pred_label = model(batch,inference=True)
                predictions.extend(pred_label.cpu().numpy())
                id.extend(batch['id'].cpu().numpy())
        # 4. dump results
        with open(self.args.test_output_csv, 'w') as f:
            f.write(f'rewire_id,label_pred\n')
            for pred_label, id in zip(self.le.inverse_transform(predictions), id):
                sample_id = 'sexism2022_english-'+str(id)
                f.write(f'{sample_id},{pred_label}\n')

    def main(self):
        setup_logging(self.args)
        setup_device(self.args)
        setup_seed(self.args)
        os.makedirs(self.args.savedmodel_path, exist_ok=True)
        if self.args.k_fold and self.args.labeltype != 'label_sexist':
            logging.critical('TODO: K-fold on Task B & Task C.')
            return
        logging.info("model parameters: %s", self.args)
        if self.args.mode=='train':
            if self.args.k_fold:
                fold_score = []
                for fold_id in range(5):
                    logging.info(f'fold: {fold_id}')
                    best_score = self.train_and_validate(fold_id=fold_id)
                    logging.info(f'fold {fold_id} best score: {best_score}')
                    fold_score.append(best_score)
                    torch.cuda.empty_cache()
                logging.info(f'5 fold avg score: {np.mean(fold_score)}')
            else:
                self.train_and_validate()
        else:
            self.inference()
        
