#!/usr/bin/env python3
import pickle
import sys
import os
import argparse
import pprint
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Print pickle file contents to a text file')
    parser.add_argument('pkl_file', help='Path to the pickle file to read')
    
    args = parser.parse_args()
    
    pkl_path = Path(args.pkl_file)
    
    if not pkl_path.exists():
        print(f"Error: File {pkl_path} does not exist")
        sys.exit(1)
    
    if not pkl_path.suffix == '.pkl':
        print(f"Error: File {pkl_path} is not a .pkl file")
        sys.exit(1)
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        txt_path = pkl_path.with_suffix('.txt')
        
        with open(txt_path, 'w') as f:
            f.write(f"Contents of {pkl_path.name}:\n")
            f.write("=" * 50 + "\n\n")
            f.write(pprint.pformat(data, indent=2, width=80))
            f.write("\n")
        
        print(f"Successfully created {txt_path}")
        
    except Exception as e:
        print(f"Error processing {pkl_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()