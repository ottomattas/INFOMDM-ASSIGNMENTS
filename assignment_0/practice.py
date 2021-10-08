'''
###!/bin/python3
### -*- encoding: utf-8 -*-
### by Antonio Hernandez Oliva, Ghislaine van den Boogerd, Otto Mättas
#######
### This script is for solving two classification problems:
### 1. Calculating impurity for a vector using gini-index as impurity measure
### 2. Calculating best split for two vectors
'''
import argparse
import sys
import numpy as np


def calculate_impurity(array):
    '''Calculate impurity for a vector using gini-index as impurity measure

        Parameters
        ----------
        array : array
            The array of binary class labels

        Raises
        ------
        RuntimeError
            Out of fuel

        Returns
        -------
        imp
            Impurity of a vector of (arbitrary length) of class labels
        '''
    # Define variable for impurity
    impurity = 0
    # Flatten array to 1d, treat all values equally
    array = array.flatten()
    # Sort array values
    array = np.sort(array)
    # Count array elements
    n_elements = array.shape[0]
    # Count 1s in array
    ones = np.count_nonzero(array)
    # Count 0s in array
    zeros = n_elements - ones
    # Probability of class 1
    p_one = ones / n_elements
    # Probability of class 0
    p_zero = zeros / n_elements
    # Calculate impurity
    impurity = p_one * p_zero
    # Return impurity
    return impurity


# Create an argument parser
parser = argparse.ArgumentParser(
    description='Calculate impurity and best split for binary class structures.'
)
# Add positional argument for source file
parser.add_argument('input',
                    type=argparse.FileType('r'),
                    help='Source file for input data')
# Add optional positional argument for destination file
parser.add_argument('output',
                    nargs='?',
                    type=argparse.FileType('w'),
                    help='OPTIONAL: Destination file for recording results')
# Parse the arguments
args = parser.parse_args()

# Import data to a numpy array
import_data = np.genfromtxt(args.input, delimiter=',', dtype=int, names=True)
# Print column names for debugging
# print(import_data.dtype.names)

# Create an array from the class column
class_data = import_data['class']

# Check if positional argument for output file is present
if args.output:
    # Open the output file object
    with open(args.output.name, 'w', encoding='utf-8') as sys.stdout:
        # Print the results
        print(f'Using file: {args.input.name}')
        print(f'Impurity is: {calculate_impurity(class_data)}')
        # Close the file object
        sys.stdout.close()
else:
    # Print the results directly to terminal
    print(f'Using file: {args.input.name}')
    print(f'Impurity is: {calculate_impurity(class_data)}')
