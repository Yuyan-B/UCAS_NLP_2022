import torch
import numpy as np
import pandas as pd
import logging,time
from utils import *
from datamanager import *
from sklearn.preprocessing import LabelEncoder
from Model import *
import os

class Run():
    def __init__(self,args):
        self.le=LabelEncoder()
        raw_data=pd.read_csv(args.train_file)
        self.le.fit(raw_data[args.labeltype])

    def train_and_validate(args):
        # 1. load data
        if not os.path.exists(f'{args.savedmodel_path}'): os.makedirs(f'{args.savedmodel_path}')
        train_dataloader, val_dataloader = create_train_dataloaders(args,self.le)
        # 2. build model and optimizers
        model = BertModel(args)
        #这里model写if载入
        optimizer, scheduler = build_optimizer(args, model)
        if args.device == 'cuda':
            model = torch.nn.parallel.DataParallel(model.to(args.device))
        # 3. training
        step = 0
        best_score = args.best_score
        start_time = time.time()
        print('len(train_dataloader): ', len(train_dataloader))
        num_total_steps = len(train_dataloader) * args.max_epochs
        #打印checkpoint值，输出整体loss
        for epoch in range(args.max_epochs):
            for i, batch in enumerate(train_dataloader):
                model.train()
                optimizer.zero_grad()
                loss, pred_label = model(batch)
                loss.backward()
                # if fgm is not None:
                #     fgm.attack()
                #     if args.use_fp16:
                #         with ac():
                #             loss_adv, _, _, _ = model(batch)
                #     else:
                #         loss_adv, _, _, _ = model(batch)
                #     loss_adv = loss_adv.mean()
                #     if args.use_fp16:
                #         scaler.scale(loss_adv).backward()
                #     else:
                #         loss_adv.backward()
                #     fgm.restore()
                # elif pgd is not None:
                #     pgd.backup_grad()
                #     for _t in range(pgd_k):
                #         pgd.attack(is_first_attack=(_t == 0))
                #         if _t != pgd_k - 1:
                #             model.zero_grad()
                #         else:
                #             pgd.restore_grad()
                #         if args.use_fp16:
                #             with ac():
                #                 loss_adv, _, _, _ = model(batch)
                #         else:
                #             loss_adv, _, _, _ = model(batch)
                #         loss_adv = loss_adv.mean()
                #         if args.use_fp16:
                #             scaler.scale(loss_adv).backward()
                #         else:
                #             loss_adv.backward()
                #     pgd.restore()
                optimizer.step()
                model.zero_grad()
                scheduler.step()
                step += 1
                if step % args.print_steps == 0:
                    time_per_step = (time.time() - start_time) / max(1, step)
                    remaining_time = time_per_step * (num_total_steps - step)
                    remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                    print_loss=loss.mean()
                    logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {print_loss:.3f}")
            # 4. validation
            loss, results = validate(model, val_dataloader)
            results = {k: round(v, 4) for k, v in results.items()}
            logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
            # 5. save checkpoint
            mean_f1 = results['f1_macro']
            if mean_f1 > best_score:
                best_score = mean_f1
                logging.info('best_score: ', best_score)
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'mean_f1': mean_f1},
                        f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1}.bin')
        
    
    def validate(model, val_dataloader):
        model.eval()
        predictions = []
        labels = []
        losses = []
        with torch.no_grad():
            for batch in val_dataloader:
                loss,pred_label = model(batch)
                loss = loss.mean()
                predictions.extend(pred_label.cpu().numpy())
                labels.extend(batch['label'].cpu().numpy())
                losses.append(loss.cpu().numpy())
        loss = np.mean(losses)
        results = evaluate(predictions, labels)
        model.train()
        return loss, results

    def inference(args):
        dataloader = create_test_dataloaders(args)
        # 2. load model
        model = BertModel(args)
        checkpoint = torch.load(args.ckpt_file, map_location='cpu')
        new_key = model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        # model.half()
        if torch.cuda.is_available():
            model = torch.nn.parallel.DataParallel(model.cuda())
        model.eval()
        # 3. inference
        predictions = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                pred_label = model(batch,inference=True)
                predictions.extend(pred_label)
        # 4. dump results
        with open(args.test_output_csv, 'w') as f:
            f.write(f'rewire_id,label_pred\n')
            for pred_label, id in zip(predictions, dataloader.id):
                pred_label=self.le.inverse_transform(predictions)
                sample_id = 'sexism2022_english-'+str(id)
                f.write(f'{sample_id},{pred_label}\n')

    def main(self,args):
        setup_logging(args)
        setup_device(args)
        setup_seed(args)
        os.makedirs(args.savedmodel_path, exist_ok=True)
        logging.info("model parameters: %s", args)

        train_and_validate(args)
        # inference(args)
        
