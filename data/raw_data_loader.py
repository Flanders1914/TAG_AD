#!/usr/bin/env python3
# python data/raw_data_loader.py --dataset cora_fixed_sbert
# python data/raw_data_loader.py --dataset citeseer_fixed_sbert
# python data/raw_data_loader.py --dataset pubmed_fixed_sbert
# python data/raw_data_loader.py --dataset arxiv_fixed_sbert

"""
Data Loader for LLMGNN Dataset Files

This module provides utilities to load and process PyTorch Geometric dataset files 
from the LLMGNN repository (https://github.com/CurryTang/LLMGNN/tree/master).
"""

import os
import torch
from typing import Dict, List
from torch_geometric.data import Data
import argparse

class LLMGNNDataLoader:
    """
    Data loader for LLMGNN datasets stored as .pt files.
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the data loader.
        
        Args:
            data_dir (str): Directory containing the .pt files (default: "raw")
        """
        self.data_dir = data_dir
        self.available_datasets = self._scan_datasets()
        
    def _scan_datasets(self) -> List[str]:
        """Scan the data directory for available .pt files."""
        if not os.path.exists(self.data_dir):
            print(f"Data directory {self.data_dir} does not exist")
            return []
        datasets = []
        for file in os.listdir(self.data_dir):
            if file.endswith('.pt'):
                dataset_name = file.replace('.pt', '')
                datasets.append(dataset_name)
        return datasets
    
    def load_dataset(self, dataset_name: str) -> Data:
        """
        Load a specific dataset.
        
        Args:
            dataset_name (str): Name of the dataset (e.g., 'cora_fixed_sbert')
            
        Returns:
            Data: PyTorch Geometric Data object containing the graph
            
        Raises:
            FileNotFoundError: If the dataset file doesn't exist
            RuntimeError: If there's an error loading the dataset
        """
        # Add .pt extension if not present
        if not dataset_name.endswith('.pt'):
            dataset_name += '.pt'
            
        file_path = os.path.join(self.data_dir, dataset_name)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        try:
            print(f"Loading dataset from {file_path}")
            data = torch.load(file_path, weights_only=False)
            
            if not isinstance(data, Data):
                raise RuntimeError(f"Expected torch_geometric.data.Data object, got {type(data)}")
                
            print(f"Successfully loaded dataset: {dataset_name}")
            return data
            
        except Exception as e:
            raise RuntimeError(f"Error loading dataset {dataset_name}: {str(e)}")
    
    def load_all_datasets(self) -> Dict[str, Data]:
        """
        Load all available datasets.
        
        Returns:
            Dict[str, Data]: Dictionary mapping dataset names to Data objects
        """
        datasets = {}
        for dataset_name in self.available_datasets:
            try:
                datasets[dataset_name] = self.load_dataset(dataset_name)
            except Exception as e:
                print(f"Failed to load dataset {dataset_name}: {str(e)}")
        return datasets
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora_fixed_sbert')
    parser.add_argument('--data_dir', type=str, default='data/raw')
    args = parser.parse_args()

    # Initialize loader
    loader = LLMGNNDataLoader(data_dir=args.data_dir)
    
    print("Available datasets:", loader.available_datasets)
    print("-"*100)

    # Load a specific dataset
    if args.dataset in loader.available_datasets:
        data = loader.load_dataset(args.dataset)
        print("Loaded dataset:")
        print(data)
        # print all label_names
        print("All label_names:")
        print(data.label_names)
        print()
        # print all attributes and their types
        print("All attributes and their types:")
        for k in data.keys():
            v = data[k]
            if isinstance(v, list):
                print(f"{k}: type: {type(v)}, length: {len(v)}, element type: {type(v[0])}")
                print(f"Example: {v[:5]}")
                print()
            elif isinstance(v, torch.Tensor):
                print(f"{k}: type: {type(v)}, shape: {v.shape}, dtype: {v.dtype}")
                print(f"Example: {v[:5]}")
                print()
            else:
                print(f"{k}: type: {type(v)}")
                print(f"Example: {v}")
                print()
    else:
        print(f"Dataset {args.dataset} not found")

    # print 5 raw text examples
    print("10 raw text examples:")
    for i in range(10):
        print(data.raw_text[i])
        print()

if __name__ == "__main__":
    main()