import numpy as np
from collections import Counter

def print_repeated_numbers(file_path):
    # Load the npz file
    npzfile = np.load(file_path)
    
    # Initialize a counter to count occurrences of each number
    counter = Counter()
    
    # Iterate over each item in the npz file
    for array_name in npzfile.files:
        array_data = npzfile[array_name]
        # Flatten the array to count each element
        flat_array = array_data.flatten()
        counter.update(flat_array)
    
    # Filter out numbers that appear more than once
    repeated_numbers = {num: count for num, count in counter.items() if count > 1}
    
    if repeated_numbers:
        # Print the numbers and their counts
        print("Numbers that appear more than once and their counts:")
        for num, count in repeated_numbers.items():
            print(f"Number: {num}, Count: {count}")
    else:
        print("No numbers appear more than once.")

# Example usage
print_repeated_numbers('output/perfect/new_banana/gaussian_binding.npz')
