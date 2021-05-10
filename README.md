# SOM-DST

## Requirements

```
python3.6
pytorch-transformers==1.0.0
torch==1.3.0a0+24ae9b5
wget==3.2
```

## Download and Preprocessing data

To download the MultiWOZ dataset and preprocess it, please run this script first.<br>
You can choose the version of the dataset. ('2.1', '2.0')<br>
The downloaded original dataset will be located in `$DOWNLOAD_PATH`, and after preprocessing, it will be located in `$TARGET_PATH`.
```
python3 create_data.py --main_dir $DOWNLOAD_PATH --target_path $TARGET_PATH --mwz_ver '2.1' # or '2.0'
```


## Model Training

To train the SOM-DST model, please run this script. <br>
`$DATASET_DIR` is the root directory of the preprocessed dataset, and `$SAVE_DIR` is output directory that best_model's checkpoint will be saved. <br>
This script contains the downloading process of pretrained-BERT checkpoint depending on `--bert_ckpt_path`. `--bert_ckpt_path` should contain either `base` or `large`. 


```
python3 train.py --data_root $DATASET_DIR --save_dir $SAVE_DIR --bert_ckpt_path `bert-base-uncased-pytorch_model.bin --op_code '4'`
```

CUDA_VISIBLE_DEVICES="0,1" nohup python train.py --data_root data/mwz2.1/ --save_dir checkpoint/ --op_code '4' >logv2.json &
### TransDST
CUDA_VISIBLE_DEVICES="0,1,2,3" nohup python TransDST_train.py --data_root data/mwz2.1/ --op_code '3-1' --batch_size 32 --grad_accumulation 1  --dec_lr 1e-4 --num_decoder_layers 4 --model TransDST >log_copy.json &


#### DDP code
##### 假设我们只用4,5,6,7号卡
CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 train_ddp.py
##### TransDST
CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 TransDST_train_ddp.py
##### 假如我们还有另外一个实验要跑，也就是同时跑两个不同实验.这时，为避免master_port冲突，我们需要指定一个新的。这里我随便敲了一个。
CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch --nproc_per_node 4 --master_port 53453 main.py

You can choose the operation set from various options via `--op_code`. The default setting is `'4'`.

```python
OP_SET = {
    '2': {'update': 0, 'carryover': 1},
    '3-1': {'update': 0, 'carryover': 1, 'dontcare': 2},
    '3-2': {'update': 0, 'carryover': 1, 'delete': 2},
    '4': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3},
    '6': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3, 'yes': 4, 'no': 5}
}
```


## Model Evaluation

If you want to evaluate the already trained model, you can run this script. <br>
`$MODEL_PATH` is the checkpoint of the model used for evaluation, and `$DATASET_DIR` is the root directory of the preprocessed dataset. <br>
You can download the pretrained SOM-DST model from [here](https://drive.google.com/file/d/1letiJzYtaul0w5xAJr7LRmjIyhYy8hiy/view?usp=sharing).

```
python3 evaluation.py --model_ckpt_path $MODEL_PATH --data_root $DATASET_DIR

python evaluation.py --model_ckpt_path checkpoint/model_best.bin --data_root data/mwz2.1
```

```
--gt_op: give the ground-truth operation for the evaluation.
--gt_p_state: give the ground-truth previous dialogue state for the evaluation.
--gt_gen: give the ground-truth generation for the evaluation.
--eval_all: evaluate all combinations of these.
```

### Main results on MultiWOZ dataset (Joint Goal Accuracy)


|Model        |MultiWOZ 2.0 |MultWOZ 2.1|
|-------------|------------|------------|
|SOM-DST Base | 51.72      | 53.01      |
|SOM-DST Large| 52.32      | 53.68      |