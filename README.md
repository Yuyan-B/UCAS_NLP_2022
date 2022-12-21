# Code for EDOS
## Document Instruction
main.py: 程序运行入口，存放参数设置  
run.py: 主体运行文件，存放训练、验证、测试函数  
datamanager.py:数据读入    
Model.py: 存放模型，包括Bert,RoBerta,TextCNN,Model_MultiTask  
utils.py：存放一些通用函数模块，包括日志设置、评价函数，MLP，PGD等  
./preprocess:存放一些数据预处理文件，包括数据分析、重采样等  
./data: 存放数据  

## Experiment
运行示例  
```
# 训练模型
python main.py --mode train --labeltype label_category --model_name textcnn --multitask --attack pgd --lambdaB 1 --lambdaC 0.3 --train_file ./data/oversampleB_train.csv --val_file ./data/oversampleB_val.csv --oversampling

# 测试模型
python main.py --mode test --labeltype label_category --model_name textcnn --multitask --attack pgd --lambdaB 1 --lambdaC 0.3 --test_file ./data/dev_task_b_entries.csv --test_output_csv ./textcnn_multitask_resultB.csv --ckpt_file ./save/model_textcnn_multitask.bin

```