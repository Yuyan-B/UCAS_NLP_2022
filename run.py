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

    def train_and_validate(self):
        # 1. load data
        if not os.path.exists(f'{self.args.savedmodel_path}'): os.makedirs(f'{self.args.savedmodel_path}')
        train_dataloader, val_dataloader = create_train_dataloaders(self.args,self.le)
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
        start_time = time.time()
        print('len(train_dataloader): ', len(train_dataloader))
        num_total_steps = len(train_dataloader) * self.args.max_epochs
        #打印checkpoint值，输出整体loss
        update=0
        for epoch in range(self.args.max_epochs):
            print('----epoch %d----' % epoch)
            if update>=self.args.early_stop:
                break
            for i, batch in enumerate(tqdm(train_dataloader)):
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
                step += 1
                if step % self.args.print_steps == 0:
                    time_per_step = (time.time() - start_time) / max(1, step)
                    remaining_time = time_per_step * (num_total_steps - step)
                    remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                    print_loss=loss.mean()
                    logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {print_loss:.3f}")
            # 4. validation
            loss, results = self.validate(model, val_dataloader)
            results = {k: round(v, 4) for k, v in results.items()}
            logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
            # 5. save checkpoint
            mean_f1 = results['f1_macro']
            update=update+1
            if mean_f1 > best_score:
                best_score = mean_f1
                logging.info(f'best_score: {best_score}')
                update=0
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'mean_f1': mean_f1},
                        f'{self.args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1}.bin')
        
    
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
        print(model.state_dict())
        checkpoint = torch.load(self.args.ckpt_file)
        print(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        print(model.state_dict())
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
        logging.info("model parameters: %s", self.args)
        if self.args.mode=='train':
            self.train_and_validate()
        else:
            self.inference()
        
