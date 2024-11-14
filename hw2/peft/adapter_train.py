import os
import logging
import sys
import numpy as np
from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
import evaluate
from transformers import RobertaModel, RobertaConfig
import torch.nn as nn
import sys
sys.path.append('/data/align-anything/boyuan/nlp-workspace/pku_nlp_2024/hw2')

from dataHelper import get_dataset  
import wandb
from statistics import mean, stdev
from datetime import datetime
from adapter import RobertaWithAdapter, Adapter




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
        name=f"{data_args.dataset_name}_{model_args.model_name_or_path}-adapter"
    )

    all_results = {"accuracy": [], "macro_f1": [], "micro_f1": []}

    for run in range(num_runs):
        logger.info(f"Run {run+1}/{num_runs} for model {model_args.model_name_or_path} on {data_args.dataset_name}")
        
        # Set seed for reproducibility
        set_seed(2024)

        # Load dataset
        raw_datasets = get_dataset(data_args.dataset_name, sep_token="<sep>")

        # Load pretrained model, tokenizer, and configuration
        num_labels = len(set(raw_datasets["train"]["label"]))

        config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        model = RobertaWithAdapter(model_args.model_name_or_path)

        # 仅微调Adapter
        for param in model.roberta.parameters():
            param.requires_grad = False

        # Tokenize dataset
        def preprocess_function(examples):
            return tokenizer(examples["text"], padding="max_length", max_length=data_args.max_seq_length, truncation=True)

        tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

        # Data collator for dynamic padding
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Define metrics for evaluation
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
         
            # Load individual metrics
            accuracy_metric = evaluate.load("accuracy")
            f1_macro_metric = evaluate.load("f1")
            f1_micro_metric = evaluate.load("f1")
            
            # Compute metrics
            accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
            f1_macro = f1_macro_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
            f1_micro = f1_micro_metric.compute(predictions=predictions, references=labels, average="micro")["f1"]
            
            return {
                "accuracy": accuracy,
                "eval_macro_f1": f1_macro,
                "eval_micro_f1": f1_micro
            }

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # Train and evaluate
        train_result = trainer.train()

        # Save the model to the custom path
        trainer.save_model(training_args.output_dir)
        
        # Evaluation
        metrics = trainer.evaluate()
        wandb.log(metrics)

        # Save results for this run
        print(metrics.keys())
        # NOTE metrics keys are eval_accuracy, eval_macro_f1, eval_micro_f1 , not aligned with return values of compute_metrics: accuracy, macro_f1, micro_f1
        all_results["accuracy"].append(metrics["eval_accuracy"])
        all_results["macro_f1"].append(metrics["eval_macro_f1"])
        all_results["micro_f1"].append(metrics["eval_micro_f1"])

        # Log metrics
        trainer.log_metrics("eval", metrics)
    # Compute averages and standard deviations
    final_results = {metric: {"mean": mean(values), "std": stdev(values)} for metric, values in all_results.items()}
    wandb.log(final_results) 

    logger.info("Final averaged results over runs:")
    logger.info(final_results)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(
            './results/models', 
            f"{model_args.model_name_or_path}_{data_args.dataset_name}_{current_time}"
        )
    os.makedirs(model_save_path, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=model_save_path,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./results/logs',
        logging_strategy="steps",
        logging_steps=10,
        report_to="wandb", 
        run_name=f"{model_args.model_name_or_path}_{data_args.dataset_name}", 
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )

    num_runs = 5
    train_and_evaluate(model_args, data_args, training_args, num_runs=num_runs)
