# An autograder for Python homework
import importlib.util
import traceback
import time
import random
exercises = {
    'flatten_list': {
        'function_name': 'flatten_list',
        'test_cases': [
            ([[1, 2, 3], [4, 5], [6]], [1, 2, 3, 4, 5, 6]),
            ([[1, 2], [3, 4]], [1, 2, 3, 4]),
            ([], []),
        ]
    },
    'char_count': {
        'function_name': 'char_count',
        'test_cases': [
            ("hello world", {'h': 1, 'e': 1, 'l': 3, 'o': 2, ' ': 1, 'w': 1, 'r': 1, 'd': 1}),
            ("aaa", {'a': 3}),
            ("", {})
        ]
    }
}

from tqdm import tqdm

def generate_large_list(size:int):
    subsize = 10
    large_list = [[i for i in range(subsize)] for _ in range(size // subsize)]
    expected_results = [i for i in range(subsize)] * (size // subsize)
    return large_list, expected_results

def generate_large_string(size):
    space_num = random.randint(0,10)
    
    large_string = "n" * (size // 3) + " "*space_num + "l" * (size // 3) + "p" * (size//3)
    expected_results = {'n': size // 3, ' ': space_num, 'l': size // 3, 'p':size//3}
    return large_string, expected_results

def load_module(file_path):
    """Dynamically load a module from a given file path."""
    spec = importlib.util.spec_from_file_location("submission", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_tests(module, exercises):
    """Run tests for all exercises and print results."""
    results = {}
    input_scales = [10**3, 10**4, 10**5, 10**6,10**7]
    
    for exercise_name, details in exercises.items():
        function_name = details['function_name']
        # test_cases = details['test_cases']
        results[exercise_name] = {'times':[],'passed': 0, 'failed': 0, 'errors': []}
        
        if not hasattr(module, function_name):
            results[exercise_name]['errors'].append(f"Function '{function_name}' not found.")
            continue
        
        function = getattr(module, function_name)
        
        for scale in tqdm(input_scales):
            print(f'Beginning of test scale: {scale}...')
            try:
                if exercise_name == 'flatten_list':
                    input_data, expected_output = generate_large_list(scale)
                elif exercise_name == 'char_count':
                    input_data, expected_output = generate_large_string(scale)
                
                start_time = time.time()
                result = function(input_data)
                end_time = time.time()

                if result == expected_output:
                    results[exercise_name]['passed'] += 1
                else:
                    results[exercise_name]['failed'] += 1
                    results[exercise_name]['errors'].append(
                        f"Scale {scale}: expected {expected_output[:10]}..., got {result[:10]}..."
                    )
                results[exercise_name]['times'].append((scale, end_time - start_time))
            except Exception as e:
                results[exercise_name]['failed'] += 1
                results[exercise_name]['errors'].append(
                    f"Error at scale {scale}: {traceback.format_exc()}"
                )
    return results

def print_results(results):
    """Print the results in a readable format."""
    for exercise_name, result in results.items():
        print(f"Exercise: {exercise_name}")
        print("  Scale * Times (input size, time in seconds):")
        for scale_time in result['times']:
            print(f"    Input size {scale_time[0]}: {scale_time[1]:.6f} seconds")
        print(f"  Passed: {result['passed']}")
        print(f"  Failed: {result['failed']}")
        if result['errors']:
            print("  Errors:")
            for error in result['errors']:
                print(f"    - {error}")
        print("\n")

if __name__ == "__main__":
    # Replace 'submission.py' with the path to the user's submission file.
    module = 'dev0' # or module = 'dev0'
    if module == 'submission':
        submission_file = 'submission.py'
    elif module == 'dev0':
        submission_file = 'submission_dev0.py'
    elif module =='dev1':
        submission_file = 'submission_dev1.py'
    user_module = load_module(submission_file)
    results = run_tests(user_module, exercises)
    print_results(results)