def flatten_list(nested_list: list):
    
    result = []
    for sublist in nested_list:
        for item in sublist:
            result.append(item)
    return result


def char_count(s: str):
    count_dict = {}
    for char in s:
        if char.islower() or char == ' ':
            if char in count_dict:
                count_dict[char] += 1
            else:
                count_dict[char] = 1
    return count_dict
