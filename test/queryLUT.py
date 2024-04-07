#!/usr/bin/env python3

import numpy as np

def load_lut_from_binary(filename):

    with open(filename, 'rb') as file:
        rows = int.from_bytes(file.read(4), byteorder='little')
        cols = int.from_bytes(file.read(4), byteorder='little')
        dtype_size = int.from_bytes(file.read(4), byteorder='little')
        
        # Assuming the LUT was saved as float32 for both x' and y' values
        dtype = np.float32
        lut = np.frombuffer(file.read(), dtype=dtype).reshape((rows, cols, 2))
        
    return lut

def query_lut(lut, x, y):

    if 0 <= y < lut.shape[0] and 0 <= x < lut.shape[1]:
        return tuple(lut[y, x])
    else:
        return None  # Out of bounds

# Example usage
filename = "../mat.bin"  # Update this with the path to your LUT file

lut = load_lut_from_binary(filename)

# Input point (you could replace these with input() for interactive use)
#x, y = 10, 20  # Example coordinates
x = int(input("Enter x coordinate: "))
y = int(input("Enter y coordinate: "))

transformed_point = query_lut(lut, x, y)
if transformed_point:
    print(f"Original point: ({x}, {y})")
    print(f"Transformed point: {transformed_point}")
else:
    print("Point is out of the bounds of the LUT.")
