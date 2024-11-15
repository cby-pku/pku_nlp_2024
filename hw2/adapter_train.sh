export WANDB_API_KEY='beaf8a16340b17e63719218d56b1faf6b6cbbf40'

models=("roberta-base")
datasets=("restaurant_sup" "acl_sup" "agnews_sup")

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        python adapter_train.py \
            --model_name_or_path "$model" \
            --dataset_name "$dataset"
    done
done