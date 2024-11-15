import os
import logging
import sys
import numpy as np
from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    DataCollatorWithPadding,
)
import evaluate

from models.roberta_with_adapter import RobertaWithAdapter  
from dataHelper import get_dataset  
import wandb
from statistics import mean, stdev
from datetime import datetime
import torch

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

@dataclass
class DataArguments:
    dataset_name: str = field(metadata={"help": "The name of the dataset to use"})
    max_seq_length: int = field(default=128, metadata={"help": "Maximum sequence length for tokenization"})

def train_and_evaluate(model_args, data_args, training_args, num_runs=5):

    wandb.init(
        project="transformer_experiment",
        name=f"adapter_{data_args.dataset_name}_{model_args.model_name_or_path}"
    )

    all_results = {"accuracy": [], "macro_f1": [], "micro_f1": []}

    for run in range(num_runs):
        logger.info(f"Run {run+1}/{num_runs} for model {model_args.model_name_or_path} on {data_args.dataset_name}")
        
        set_seed(2024 + run)

        # load dataset
        raw_datasets = get_dataset(data_args.dataset_name, sep_token="<sep>")

        # load model config and tokenizer
        num_labels = len(set(raw_datasets["train"]["label"]))

        config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

        # init model
        model = RobertaWithAdapter.from_pretrained(model_args.model_name_or_path, config=config)

        # freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # only unfreeze adapter parameters
        for adapter in model.adapters:
            for param in adapter.parameters():
                param.requires_grad = True

        # record trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params} / {total_params}")

        # move model to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # tokenize dataset
        def preprocess_function(examples):
            return tokenizer(examples["text"], padding="max_length", max_length=data_args.max_seq_length, truncation=True)

        tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

        # data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # compute metrics
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
         
            # load metrics
            accuracy_metric = evaluate.load("accuracy")
            f1_macro_metric = evaluate.load("f1")
            f1_micro_metric = evaluate.load("f1")
            
            # compute metrics
            accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
            f1_macro = f1_macro_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
            f1_micro = f1_micro_metric.compute(predictions=predictions, references=labels, average="micro")["f1"]
            
            return {
                "accuracy": accuracy,
                "eval_macro_f1": f1_macro,
                "eval_micro_f1": f1_micro
            }

        # init trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # train and evaluate
        train_result = trainer.train()

        # save model
        trainer.save_model(training_args.output_dir)
        
        # evaluate
        metrics = trainer.evaluate()
        wandb.log(metrics)

        # save this run's results
        logger.info(f"Metrics keys: {metrics.keys()}")
        all_results["accuracy"].append(metrics.get("eval_accuracy", 0))
        all_results["macro_f1"].append(metrics.get("eval_macro_f1", 0))
        all_results["micro_f1"].append(metrics.get("eval_micro_f1", 0))

        # record this run's results
        trainer.log_metrics("eval", metrics)
    # calculate mean and standard deviation
    final_results = {metric: {"mean": mean(values), "std": stdev(values)} for metric, values in all_results.items()}
    wandb.log(final_results) 

    logger.info("Final averaged results over runs:")
    logger.info(final_results)

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(
            './results/models-adapter', 
            f"{model_args.model_name_or_path}_{data_args.dataset_name}_{current_time}"
        )
    os.makedirs(model_save_path, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=model_save_path,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-4 ,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./results/logs',
        logging_strategy="steps",
        lr_scheduler_type="linear",
        logging_steps=10,
        report_to="wandb", 
        run_name=f"adapter_{model_args.model_name_or_path}_{data_args.dataset_name}", 
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        
    )

    num_runs = 5
    train_and_evaluate(model_args, data_args, training_args, num_runs=num_runs)
