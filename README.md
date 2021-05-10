# TransDST

## Requirements

```
python3.6
transformers==4.5.0
torch==1.4
```

## Download and Preprocessing data

To download the MultiWOZ dataset and preprocess it, please run this script first.<br>
You can choose the version of the dataset. ('2.1', '2.0')<br>
The downloaded original dataset will be located in `$DOWNLOAD_PATH`, and after preprocessing, it will be located in `$TARGET_PATH`.
```
python3 create_data.py --main_dir $DOWNLOAD_PATH --target_path $TARGET_PATH --mwz_ver '2.1' # or '2.0'
```


## Model Training

`$DATASET_DIR` is the root directory of the preprocessed dataset, and `$SAVE_DIR` is output directory that best_model's checkpoint will be saved. <br>

### TransDST

```
bash run.sh
```

#### DDP code
##### TransDST
```
CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 TransDST_train_ddp.py
```

##### 假如我们还有另外一个实验要跑，也就是同时跑两个不同实验.这时，为避免master_port冲突，我们需要指定一个新的。这里我随便敲了一个。

```
CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch --nproc_per_node 4 --master_port 53453 main.py
```

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

### Main results on MultiWOZ dataset (Joint Goal Accuracy)


|Model          |MultiWOZ 2.0|MultWOZ 2.1|
|---------------|------------|------------|
|SOM-DST(Base)  | 51.72      | 53.01      |
|SOM-DST(Large) | 52.32      | 53.68      |
|TransDST(Base) |            | 53.91      |