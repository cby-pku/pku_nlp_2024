def flatten_list(nested_list: list):
    
    return [item for sublist in nested_list for item in sublist]


def char_count(s: str):
    return {char: s.count(char) for char in s if char.islower() or char == ' '}
