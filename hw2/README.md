# HW2: Datasets \ Training & PEFT

## Dataset Processing
You should download raw dataset and process the dataset as below.
For debug, you can use the following command to process the dataset and check the data statistics.
```
python dataHelper.py
```

## Training
You can export your own WANDB_API_KEY in the train.sh file.
```
export WANDB_API_KEY=<your_wandb_api_key>   
```

Then, you can train the model with the following command.
```
bash train.sh
```

Now, we have 3 models and 3 datasets, so there should be 9 runs in the Weights & Biases.

| Models                  | Datasets        |
|-------------------------|-----------------|
| roberta-base            | restaurant_sup  |
| bert-base-uncased       | acl_sup         |
| allenai/scibert_scivocab_uncased | agnews_sup      |

For reproducibility, you can change the hyperparameters in the train.py file. Current hyperparameters are:
- learning_rate: 2e-5
- per_device_train_batch_size: 8
- per_device_eval_batch_size: 8
- num_train_epochs: 3
- weight_decay: 0.01
- logging_steps: 10

We use the accuracy, macro_f1, micro_f1 as the evaluation metrics and will use 5 runs to average and calculate the standard deviation.


## PEFT
You can train the adapter with the following command.
```
cd peft
bash adapter_train.sh
```

In adapter_train.sh, we use the following hyperparameters:
- learning_rate: 5e-4
- per_device_train_batch_size: 16
- per_device_eval_batch_size: 16
- num_train_epochs: 3
- weight_decay: 0.01
- logging_steps: 10
- adapter_dim: 64

In adapter.py, we inherit the RobertaModel and add the adapter structure to the model.