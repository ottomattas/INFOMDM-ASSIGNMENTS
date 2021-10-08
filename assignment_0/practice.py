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


def calculate_impurity(array_y):
    '''Calculate impurity for a vector using gini-index as impurity measure

        Parameters
        ----------
        array_y : array
            The array of binary class labels

        Returns
        -------
        impurity
            Impurity of a vector (of arbitrary length) of class labels
        ny_elements
            Number of elements in the full array
        '''
    # Define variable for impurity
    impurity = 0
    # Flatten array to 1d, treat all values equally
    array_y = array_y.flatten()
    # Count array elements
    ny_elements = array_y.shape[0]
    # Count 1s in array
    ones = np.count_nonzero(array_y)
    # Probability of class 1
    p_one = ones / ny_elements
    # Probability of class 0
    p_zero = 1 - (ones / ny_elements)
    # Calculate impurity
    impurity = p_one * p_zero
    # Print and return impurity and the number of elements
    print('Impurity is: ', impurity)
    return impurity, ny_elements


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
