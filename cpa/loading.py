# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 20:34:34 2024

@author: jlu
"""

import re
import os
import numpy as np

def parse_fn_zero_num(filename):
    pattern = r"N=(\d+)_M=(\d+)_.*?kappa_([\d.]+)_.*?EIm([\d]+(?:\.\d+)?)(?=\D|$)"
    par = re.findall(pattern, filename)[0]
    if par:
        return {
            "N": int(par[0]),
            "M": int(par[1]),
            "kappa": float(par[2]),
            "EIm": float(par[3])
        }
    else:
        return None
    
def load_zero_num(directory, input_params):
    loaded_arrays = np.array([])
    for file_name in os.listdir(directory):
        if file_name.endswith('.npy'):
            params = parse_fn_zero_num(file_name)
            if params:
                matches = all(params.get(key) == input_params.get(key) for key in ['M', 'N', 'kappa', "EIm"])
                if matches:
                    file_path = os.path.join(directory, file_name)
                    data = np.load(file_path, allow_pickle=True, fix_imports=True)
                    loaded_arrays = np.append(loaded_arrays, data)
    return loaded_arrays


def parse_fn_dynamics(filename):
    pattern = r"N=(\d+)_M=(\d+)_kappa_([\d.]+)"
    par = re.findall(pattern, filename)[0]
    if par:
        return {
            "N": int(par[0]),
            "M": int(par[1]),
            "kappa": float(par[2])
        }
    else:
        return None
    
def load_dynamics(directory, input_params):
    loaded_arrays = None  # Initialize as None to start with
    for file_name in os.listdir(directory):
        if file_name.startswith('info_'):
            continue  # Skip files starting with 'info_'
        if file_name.endswith('.npy'):
            # print(file_name)
            params = parse_fn_dynamics(file_name)
            if params:
                matches = all(params.get(key) == input_params.get(key) for key in ['M', 'N', 'kappa'] if key != 'width')
                if matches:
                    file_path = os.path.join(directory, file_name)
                    data = np.load(file_path, allow_pickle=True, fix_imports=True)
                    if loaded_arrays is None:
                        loaded_arrays = data  # First valid array, assign it to loaded_arrays
                    else:
                        # Ensure data has the same number of rows as loaded_arrays
                        if loaded_arrays.shape[0] == data.shape[0]:
                            loaded_arrays = np.hstack((loaded_arrays, data))
                        else:
                            raise ValueError("Arrays have mismatched number of rows and cannot be stacked horizontally.")
    return loaded_arrays

def load_dynamics_lambda_info(directory, input_params):
    loaded_arrays = None  # Initialize as None to start with
    for file_name in os.listdir(directory):
        if file_name.endswith('.npy') & file_name.startswith('info_'):
            # print(file_name)
            params = parse_fn_dynamics(file_name)
            if params:
                matches = all(params.get(key) == input_params.get(key) for key in input_params if key != 'width')
                if matches:
                    file_path = os.path.join(directory, file_name)
                    data = np.load(file_path, allow_pickle=True, fix_imports=True)
                    data = np.transpose([np.array(i) for i in data[:,-1]])
                    if loaded_arrays is None:
                        loaded_arrays = data  # First valid array, assign it to loaded_arrays
                    else:
                        # Ensure data has the same number of rows as loaded_arrays
                        if loaded_arrays.shape[0] == data.shape[0]:
                            loaded_arrays = np.hstack((loaded_arrays, data))
                        else:
                            raise ValueError("Arrays have mismatched number of rows and cannot be stacked horizontally.")
    return loaded_arrays

def parse_fn_Ta(filename):
    pattern = r"N=(\d+)_M=(\d+)_.*?kappa_([\d.]+)_.*?EIm([\d.]+)(?:\.npy)$"
    par = re.findall(pattern, filename)[0]
    if par:
        return {
            "N": int(par[0]),
            "M": int(par[1]),
            "kappa": float(par[2]),
            "EIm": float(par[3])
        }
    else:
        return None
    
def load_Ta(directory, input_params):
    loaded_arrays = np.array([])
    for file_name in os.listdir(directory):
        if file_name.startswith('Ta') and file_name.endswith('.npy'):
            params = parse_fn_Ta(file_name)
            if params:
                matches = all(params[key] == input_params[key] for key in ['M', 'N', 'kappa', "EIm"])
                if matches:
                    file_path = os.path.join(directory, file_name)
                    data = np.load(file_path, allow_pickle=True, fix_imports=True)
                    loaded_arrays = np.append(loaded_arrays, data)
    return loaded_arrays

def parse_fn_ei_Heff(filename):
    pattern = r"M(\d+)_N(\d+)_.*?kappa(\d+(?:\.\d+)?)(?=\D|$)"

    par = re.findall(pattern, filename)[0]
    if par:
        return {
            "M": int(par[0]),
            "N": int(par[1]),
            "kappa": float(par[2])
        }
    else:
        return None
    
def load_ei_Heff(directory, input_params):
    loaded_arrays = np.array([])
    for file_name in os.listdir(directory):
        if file_name.endswith('.npy'):
            # print(file_name)
            params = parse_fn_ei_Heff(file_name)
            if params:
                matches = all(params.get(key) == input_params.get(key) for key in ['M', 'N', 'kappa'])
                if matches:
                    file_path = os.path.join(directory, file_name)
                    data = np.load(file_path, allow_pickle=True, fix_imports=True)
                    loaded_arrays = np.append(loaded_arrays, data)
    return loaded_arrays