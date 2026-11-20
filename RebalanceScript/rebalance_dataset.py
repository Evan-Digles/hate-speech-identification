"""
Dataset Rebalancing Script
Rebalances the hate speech dataset to have ~4x more Hate Speech samples
Processes large files in chunks to avoid memory issues
"""

import pandas as pd
import numpy as np
from sklearn.utils import resample
import argparse
import os


def analyze_dataset(input_file, chunksize=10000):
    """
    Analyze the dataset class distribution
    
    Args:
        input_file: Path to input CSV file
        chunksize: Size of chunks to read (for large files)
    
    Returns:
        Dictionary with class counts
    """
    print(f"Analyzing dataset: {input_file}")
    
    # Try to read the entire file first
    try:
        df = pd.read_csv(input_file)
        class_counts = df['class'].value_counts().to_dict()
        total_samples = len(df)
        print(f"\nTotal samples: {total_samples}")
        print(f"Class distribution:")
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            percentage = (count / total_samples) * 100
            class_name = get_class_name(class_id)
            print(f"  Class {class_id} ({class_name}): {count:6d} ({percentage:5.2f}%)")
        return class_counts, df
    except Exception as e:
        print(f"Cannot read entire file into memory: {e}")
        print("Processing in chunks...")
        
        # Process in chunks
        class_counts = {0: 0, 1: 0, 2: 0}
        total_samples = 0
        
        for chunk in pd.read_csv(input_file, chunksize=chunksize):
            chunk_counts = chunk['class'].value_counts().to_dict()
            for class_id, count in chunk_counts.items():
                class_counts[class_id] = class_counts.get(class_id, 0) + count
            total_samples += len(chunk)
        
        print(f"\nTotal samples: {total_samples}")
        print(f"Class distribution:")
        for class_id in sorted(class_counts.keys()):
            count = class_counts.get(class_id, 0)
            percentage = (count / total_samples) * 100
            class_name = get_class_name(class_id)
            print(f"  Class {class_id} ({class_name}): {count:6d} ({percentage:5.2f}%)")
        
        return class_counts, None


def get_class_name(class_id):
    """Get class name from ID"""
    class_names = {
        0: 'Hate Speech',
        1: 'Offensive Speech',
        2: 'Normal Speech'
    }
    return class_names.get(class_id, 'Unknown')


def rebalance_dataset(input_file, output_file, hate_multiplier=4.0, chunksize=10000):
    """
    Rebalance dataset by oversampling Hate Speech class
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        hate_multiplier: How many times to multiply Hate Speech samples (default: 4.0)
        chunksize: Size of chunks for large files
    """
    print(f"\n{'='*70}")
    print(f"REBALANCING DATASET")
    print(f"{'='*70}")
    
    # Analyze original dataset
    class_counts, df = analyze_dataset(input_file, chunksize)
    
    if df is not None:
        # Can process entire dataset in memory
        print(f"\nProcessing entire dataset in memory...")
        rebalanced_df = rebalance_in_memory(df, hate_multiplier)
        
        # Save to output file
        print(f"\nSaving rebalanced dataset to: {output_file}")
        rebalanced_df.to_csv(output_file, index=False)
        
        # Analyze rebalanced dataset
        print(f"\n{'='*70}")
        print(f"REBALANCED DATASET STATISTICS")
        print(f"{'='*70}")
        new_class_counts = rebalanced_df['class'].value_counts().to_dict()
        total_samples = len(rebalanced_df)
        print(f"Total samples: {total_samples}")
        print(f"Class distribution:")
        for class_id in sorted(new_class_counts.keys()):
            count = new_class_counts[class_id]
            percentage = (count / total_samples) * 100
            class_name = get_class_name(class_id)
            original_count = class_counts.get(class_id, 0)
            increase = count - original_count
            print(f"  Class {class_id} ({class_name}): {count:6d} ({percentage:5.2f}%) [+{increase} samples]")
        
        print(f"\n✓ Rebalanced dataset saved successfully!")
        print(f"  Original samples: {sum(class_counts.values())}")
        print(f"  New total: {total_samples}")
        print(f"  Hate Speech increased by {hate_multiplier}x")
        
    else:
        # Process in chunks
        print(f"\nProcessing large dataset in chunks...")
        rebalance_in_chunks(input_file, output_file, class_counts, hate_multiplier, chunksize)


def rebalance_in_memory(df, hate_multiplier):
    """
    Rebalance dataset that fits in memory
    
    Args:
        df: Original DataFrame
        hate_multiplier: Multiplier for Hate Speech class
    
    Returns:
        Rebalanced DataFrame
    """
    # Separate by class
    hate_speech = df[df['class'] == 0]
    offensive_speech = df[df['class'] == 1]
    normal_speech = df[df['class'] == 2]
    
    # Calculate target number of Hate Speech samples
    original_hate_count = len(hate_speech)
    target_hate_count = int(original_hate_count * hate_multiplier)
    
    print(f"\nOversampling Hate Speech:")
    print(f"  Original: {original_hate_count} samples")
    print(f"  Target: {target_hate_count} samples ({hate_multiplier}x)")
    
    # Oversample Hate Speech with replacement
    hate_speech_resampled = resample(
        hate_speech,
        n_samples=target_hate_count,
        replace=True,
        random_state=42
    )
    
    # Combine all classes
    rebalanced_df = pd.concat([
        hate_speech_resampled,
        offensive_speech,
        normal_speech
    ], ignore_index=True)
    
    # Shuffle the dataset
    rebalanced_df = rebalanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return rebalanced_df


def rebalance_in_chunks(input_file, output_file, class_counts, hate_multiplier, chunksize):
    """
    Rebalance large dataset by processing in chunks
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        class_counts: Dictionary with original class counts
        hate_multiplier: Multiplier for Hate Speech class
        chunksize: Size of chunks to process
    """
    # Calculate how many times to duplicate Hate Speech samples
    original_hate_count = class_counts.get(0, 0)
    target_hate_count = int(original_hate_count * hate_multiplier)
    additional_samples_needed = target_hate_count - original_hate_count
    
    print(f"\nOversampling Hate Speech:")
    print(f"  Original: {original_hate_count} samples")
    print(f"  Target: {target_hate_count} samples ({hate_multiplier}x)")
    print(f"  Need {additional_samples_needed} additional samples")
    
    # First pass: Write all original data
    print("\nPass 1: Writing original data...")
    first_chunk = True
    hate_speech_samples = []
    
    for chunk in pd.read_csv(input_file, chunksize=chunksize):
        # Save all data
        if first_chunk:
            chunk.to_csv(output_file, index=False, mode='w')
            first_chunk = False
        else:
            chunk.to_csv(output_file, index=False, mode='a', header=False)
        
        # Collect Hate Speech samples for resampling
        hate_samples = chunk[chunk['class'] == 0]
        if len(hate_samples) > 0:
            hate_speech_samples.append(hate_samples)
    
    # Combine all Hate Speech samples
    print("\nPass 2: Oversampling Hate Speech class...")
    all_hate_speech = pd.concat(hate_speech_samples, ignore_index=True)
    
    # Resample to get additional samples
    additional_hate_speech = resample(
        all_hate_speech,
        n_samples=additional_samples_needed,
        replace=True,
        random_state=42
    )
    
    # Append additional Hate Speech samples
    additional_hate_speech.to_csv(output_file, index=False, mode='a', header=False)
    
    print(f"\n✓ Rebalanced dataset saved to: {output_file}")
    
    # Analyze final dataset
    print(f"\nAnalyzing final dataset...")
    final_counts, _ = analyze_dataset(output_file, chunksize)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Rebalance hate speech dataset by oversampling Hate Speech class'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='Dataset-Mini.csv',
        help='Input dataset file (default: Dataset-Mini.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='Dataset-Rebalanced.csv',
        help='Output dataset file (default: Dataset-Rebalanced.csv)'
    )
    parser.add_argument(
        '--multiplier',
        type=float,
        default=4.0,
        help='Multiplier for Hate Speech class (default: 4.0 = 4x more samples)'
    )
    parser.add_argument(
        '--chunksize',
        type=int,
        default=10000,
        help='Chunk size for processing large files (default: 10000)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        return
    
    # Rebalance dataset
    rebalance_dataset(
        input_file=args.input,
        output_file=args.output,
        hate_multiplier=args.multiplier,
        chunksize=args.chunksize
    )


if __name__ == "__main__":
    main()
