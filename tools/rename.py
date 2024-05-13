# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 20:07:43 2024

@author: jlu
"""

import os

def rename_files_in_subfolders(top_directory):
    # Walk through all directories and files in the directory
    for dirpath, dirnames, filenames in os.walk(top_directory):
        for filename in filenames:
            if filename.startswith('Rmt_EigValues'):
                # Construct the new filename
                new_name = filename.replace('Rmt_EigValues', 'Smat', 1)  # Replace the prefix
                new_name = new_name.replace('S_full_', '')  # Remove 'S_full_'
                
                # Construct full file paths
                old_file = os.path.join(dirpath, filename)
                new_file = os.path.join(dirpath, new_name)
                
                # Rename the file
                os.rename(old_file, new_file)
                print(f"Renamed '{old_file}' to '{new_file}'")
            if filename.startswith('Zero_Rmt_EigValues'):
                # Construct the new filename
                new_name = filename.replace('Zero_Rmt_EigValues', 'Zero_S', 1)  # Replace the prefix
                new_name = new_name.replace('S_full_', '')  # Remove 'S_full_'
                new_name = new_name.replace('_update1', '')  # Remove 'S_full_'
                
                # Construct full file paths
                old_file = os.path.join(dirpath, filename)
                new_file = os.path.join(dirpath, new_name)
                
                # Rename the file
                os.rename(old_file, new_file)
                print(f"Renamed '{old_file}' to '{new_file}'")

    
# Specify the directory containing your files
directory_path = 'd:/onedrive/OneDrive - Université Nice Sophia Antipolis/Nice/2021-11-03-RMT_Zeros-extract/data/data_zero/'
rename_files_in_subfolders(directory_path)