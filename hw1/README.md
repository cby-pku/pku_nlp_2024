# HW1: Python, RNN & CNN

### Task 1: Python Basics


### Notes
`autograder.py` mainly for evaluation of the code correctness
I have changed for a new version, namely `autograder_plus.py` for large-scale input experiment and difference implement methods comparison.


### Details

We implemented three different approaches to solve the problems, corresponding to the code in `submission.py`, `submission_dev0.py`, and `submission_dev1.py`. The specific implementations are as follows:

- **Task 1: Flatten a list of lists**
  - **Approach A: Using List Comprehension.** This approach iterates through each sublist, then through each item in the sublist, flattening the list into a single list.
  
  - **Approach B: Using a `for` loop.** This method iterates over the sublists and appends each item to a result list.
  
  - **Approach C: Recursion.** We recursively flatten the list, which can handle deeply nested structures.

- **Task 2: Character count (only for lowercase).**
  - **Approach A: Using `str.count`.** This method directly calls `str.count` inside a dictionary comprehension to count occurrences of each character.
  
  - **Approach B: Using a `for` loop and `Counter`.** We utilize Python's `Counter` to count characters, then filter out the non-lowercase ones.
  
  - **Approach C: Only using a `for` loop.** A manual counting method that iterates through the string and counts each character without using any external libraries.
