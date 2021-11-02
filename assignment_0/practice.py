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


def calculate_splitpoint_candidates(array_x):
    '''Calculate split candidates for two vectors using gini-index as impurity measure

        Parameters
        ----------
        array_x : array
            The array of numerical attributes

        Returns
        -------
        splitpoint_candidates
            Splitpoint candidates of a vector (of arbitrary length) of numerical values
        ny_elements
            Number of elements in the full array
        '''
    # Define array for splitpoint candidates
    splitpoint_candidates = []
    # Flatten array to 1d, treat all values equally
    array_x = array_x.flatten()
    # Count array elements
    nx_elements = array_x.shape[0]
    # Select and sort unique numerical values
    unique_x, _ = np.unique(array_x, return_counts=True)
    # Calculate candidate splitpoins
    splitpoint_candidates = (np.delete(unique_x, -1) +
                             np.delete(unique_x, 0)) / 2
    # Print and return splitpoint candidates and the number of elements
    print('Splitpoint candidates are: ', splitpoint_candidates)
    print('\n')
    return splitpoint_candidates, nx_elements


def calculate_best_split(array_x, array_y):
    '''Calculate best split for two vectors using gini-index as impurity measure

        Parameters
        ----------
        array_x : array
            The array of numerical attributes
        array_y : array
            The array of binary class labels

        Returns
        -------
        bestsplit
            The best split of two vectors  (of arbitrary length);
            Best split is the split that maximizes the impurity reduction
        '''
    # Define variable for maximum impurity reduction
    impurity_reduction_maximum = 0
    # Calculate parent node impurity and get element count
    parent_impurity, n_elements = calculate_impurity(array_y)
    # Calculate splitpoint candidates
    splitpoint_candidates, _ = calculate_splitpoint_candidates(array_x)
    # For each candidate splitpoint
    for elem in splitpoint_candidates:
        # Print candidate splitpoint for debugging
        print('Splitpoint candidate: ', elem)
        # Define and reset variable for impurity reduction
        impurity_reduction = 0
        # Check which elements to consider for both child nodes
        consider_left = array_x <= elem
        consider_right = array_x > elem

        # # Create array for both child nodes for debugging
        # values_left = array_x[consider_left]
        # values_right = array_x[consider_right]
        # # Print child node arrays for debugging
        # print('Child node on the left contains: ', values_left)
        # print('Child node on the right contains: ', values_right)

        # Calculate weights for both child nodes
        weight_left = array_x[consider_left].shape[0] / n_elements
        weight_right = array_x[consider_right].shape[0] / n_elements
        # Calculate child node impurities
        child_impurity_left, _ = calculate_impurity(array_y[consider_left])
        child_impurity_right, _ = calculate_impurity(array_y[consider_right])
        # Calculate impurity reduction
        impurity_reduction = parent_impurity - (
            (weight_left * child_impurity_left) +
            (weight_right * child_impurity_right))
        # Find maximum impurity reduction
        impurity_reduction_maximum = max(impurity_reduction_maximum,
                                         impurity_reduction)
        # Find best split based on maximum impurity reduction
        if impurity_reduction == impurity_reduction_maximum:
            best_split = elem
        # Print empty line for visual clarity
        print('\n')
    # Print and return the best split
    print('Best split is: ', best_split)
    return best_split


def get_args(argv=None):
    '''Create an argument parser

        Parameters
        ----------
        argv : list
            List of arguments to pass to the parser

        Returns
        -------
        parser.parse_args(argv)
            The parser object
        '''
    # Create an object for argument parser
    parser = argparse.ArgumentParser(
        description=
        'Calculate impurity and best split for binary class structures.',
        formatter_class=argparse.RawTextHelpFormatter)
    # Add positional argument for source file
    parser.add_argument('input',
                        type=argparse.FileType('r'),
                        help='Source file for input data')
    # Add optional positional argument for destination file
    parser.add_argument(
        'output',
        nargs='?',
        type=argparse.FileType('w'),
        help='OPTIONAL: Destination file for recording results')
    # Add positional argument for source file
    parser.add_argument(
        '-c',
        '--column',
        action='store',
        dest='column_name',
        default='income',
        help='Numeric field you would like to use for splitting  \n' +
        'Defaults to "income"')
    # Return the parser
    return parser.parse_args(argv)


def main():
    '''Initialise the program

        Parameters
        ----------
        args : obj
            Argument parser object
        import_data : array
            Import data for the numpy array
        class_data : array
            Binary class information from import data
        numerical_data : array
            Optional: numerical values from import data
        '''
    # Create argument parser
    args = get_args()

    # Import data to a numpy array
    import_data = np.genfromtxt(args.input,
                                delimiter=',',
                                dtype=int,
                                names=True)
    # # Print column names for debugging
    # print(import_data.dtype.names)

    # Create an array from the class column
    class_data = import_data['class']

    # Check if positional argument for output file is present
    if args.output:
        # Open the output file object
        with open(args.output.name, 'w', encoding='utf-8') as sys.stdout:
            # Print the results
            print(f'Input data: {args.input.name}')
            # Check for numerical data
            try:
                numerical_data = import_data[args.column_name]
                calculate_best_split(numerical_data, class_data)
            except ValueError:
                print('Numerical data not present')
                calculate_impurity(class_data)
            # Close the file object
            sys.stdout.close()
    else:
        # Print the results directly to terminal
        print(f'Input data: {args.input.name}')
        # Check for numerical data
        try:
            numerical_data = import_data[args.column_name]
            calculate_best_split(numerical_data, class_data)
        except ValueError:
            print('Numerical data not present')
            calculate_impurity(class_data)


# Execute
if __name__ == "__main__":
    main()
