def flatten_list(nested_list: list):
    
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


from collections import Counter

def char_count(s: str):
    counter = Counter(s)
    return {char: count for char, count in counter.items() if char.islower() or char == ' '}
