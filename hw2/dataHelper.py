import json
from datasets import Dataset, DatasetDict, load_dataset
from collections import Counter
from tools import logger

def get_dataset(dataset_name, sep_token):
    '''
    dataset_name: str
    sep_token: str
    supported_dataset: restaurant_sup, laptop_sup, acl_sup, agnews_sup
    '''
    if dataset_name == 'restaurant_sup':
        dataset = prepare_restaurant_sup(sep_token)
    elif dataset_name == 'laptop_sup':
        dataset = prepare_laptop_sup(sep_token)
    elif dataset_name == 'acl_sup':
        dataset = prepare_acl_sup(sep_token)
    elif dataset_name == 'agnews_sup':
        dataset = prepare_agnews_sup()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return dataset

def prepare_restaurant_sup(sep_token):
    train_texts, train_labels = load_absa_data('./dataset/SemEval14-res/train.json', sep_token)
    test_texts, test_labels = load_absa_data('./dataset/SemEval14-res/test.json', sep_token)
    
    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})
    
    return DatasetDict({'train': train_dataset, 'test': test_dataset})

def prepare_laptop_sup(sep_token):
    train_texts, train_labels = load_absa_data('./dataset/SemEval14-laptop/train.json', sep_token)
    test_texts, test_labels = load_absa_data('./dataset/SemEval14-laptop/test.json', sep_token)
    
    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})
    
    return DatasetDict({'train': train_dataset, 'test': test_dataset})

def prepare_acl_sup(sep_token):
    train_texts, train_labels = load_acl_data('./dataset/acl_sup/train.jsonl', sep_token)
    test_texts, test_labels = load_acl_data('./dataset/acl_sup/test.jsonl', sep_token)
    
    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})
    
    return DatasetDict({'train': train_dataset, 'test': test_dataset})

def prepare_agnews_sup():

    dataset = load_dataset('ag_news', split='test')
    
    train_test_split = dataset.train_test_split(test_size=0.1, seed=2022)

    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']
    
    return DatasetDict({'train': train_dataset, 'test': test_dataset})

def load_absa_data(file_path, sep_token):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    texts = []
    labels = []
    polarity_map = {'positive': 0, 'neutral': 1, 'negative': 2}
    
    for item in data.values():
        term = item['term']
        sentence = item['sentence']
        polarity = item['polarity']
        
        text = f"{term} {sep_token} {sentence}"
        label = polarity_map[polarity]
        
        texts.append(text)
        labels.append(label)
    
    return texts, labels

def load_acl_data(file_path, sep_token):
    texts = []
    labels = []
    
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            text = item['text']
            label = item['label']
            
            texts.append(text)
            labels.append(label)
    
    return texts, labels

def prepare_few_shot_dataset(dataset, num_shots, seed=42):
    """
    num_shots: int
    """
    import random
    random.seed(seed)
    
    few_shot_train = []
    label_to_samples = {}
    
    for example in dataset['train']:
        label = example['label']
        if label not in label_to_samples:
            label_to_samples[label] = []
        label_to_samples[label].append(example)
    
    # NOTE Sample num_shots examples per class
    for label, samples in label_to_samples.items():
        few_shot_train.extend(random.sample(samples, min(num_shots, len(samples))))
    
    few_shot_train_dataset = Dataset.from_list(few_shot_train)
    
    return DatasetDict({'train': few_shot_train_dataset, 'test': dataset['test']})

def get_few_shot_dataset(dataset_name, sep_token, num_shots):
    """
    num_shots: int, number of samples per class
    """
    
    dataset = get_dataset(dataset_name, sep_token)
    return prepare_few_shot_dataset(dataset, num_shots)


def debug_dataset(dataset_name, sep_token, shot_type, num_shots=None):
    """
    shot_type: str, 'zero_shot' or 'few_shot'
    num_shots: int, number of samples per class (only for few_shot)
    """
    logger(f"{shot_type.capitalize()} Debug: {dataset_name}")

    if shot_type == 'zero_shot':
        dataset = get_dataset(dataset_name, sep_token)
    elif shot_type == 'few_shot':
        if num_shots is None:
            raise ValueError("num_shots must be provided for few_shot")
        dataset = get_few_shot_dataset(dataset_name.replace('_fs', '_sup'), sep_token, num_shots)
    else:
        raise ValueError("Invalid shot_type. Choose 'zero_shot' or 'few_shot'.")

    print(dataset['train'][0])
    print(len(dataset['train']))

    labels = [example['label'] for example in dataset['train']]

    label_distribution = Counter(labels)
    print("Label distribution:", label_distribution)
    
    
if __name__ == '__main__':
    zero_shot_dataset_list = ['restaurant_sup', 'laptop_sup', 'acl_sup', 'agnews_sup']
    few_shot_dataset_list = ['restaurant_fs', 'laptop_fs', 'acl_fs', 'agnews_fs']
    sep_token = '<sep>'
    for dataset_name in zero_shot_dataset_list:
        debug_dataset(dataset_name, sep_token, 'zero_shot')
    for dataset_name in few_shot_dataset_list:
        debug_dataset(dataset_name, sep_token, 'few_shot', num_shots=5)
