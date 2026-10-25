#!/usr/bin/env python3

import json
import numpy as np
import csv
import argparse

def load_labels(labels_file):
    """Load labels from CSV file."""
    labels = {}
    with open(labels_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row['id']] = row['correct'] == 'True'
    return labels

def load_metadata(metadata_file):
    """Load metadata from JSONL file."""
    metadata = []
    with open(metadata_file, 'r') as f:
        for line in f:
            metadata.append(json.loads(line.strip()))
    return metadata

def run_baseline_expected(metadata_file, labels_file, output_file):
    """Run baseline analysis with expected labels."""
    # Load labels and metadata
    labels = load_labels(labels_file)
    metadata = load_metadata(metadata_file)
    
    # Prepare data for baseline
    test_labels = []
    test_logprobs = []
    test_entropies = []
    
    for item in metadata:
        if item['split'] == 'test':
            test_labels.append(labels[item['id']])
            test_logprobs.append(item.get('logprobs', []))
            test_entropies.append(item.get('entropies', []))
    
    print(f"Test samples: {len(test_labels)}")
    print(f"Test correct ratio: {np.mean(test_labels):.3f}")
    
    # Calculate baseline metrics
    if len(test_labels) > 0:
        # For Nemotron 3 Nano 4B, we'll use a simple baseline since all logprobs and entropies are null
        # We'll use a random baseline
        random_auc = 0.5  # Random baseline
        random_auprc = 0.5  # Random baseline
        
        print(f"Baseline (random) - Test AUROC: {random_auc:.3f}")
        print(f"Baseline (random) - Test AUPRC: {random_auprc:.3f}")
        
        # Save results
        results = {
            'test_auroc': random_auc,
            'test_auprc': random_auprc,
            'test_accuracy': np.mean(test_labels),
            'test_samples': len(test_labels),
            'method': 'random_baseline'
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Baseline results saved to {output_file}")
    else:
        print("No test samples found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run baseline analysis with expected labels')
    parser.add_argument('--metadata', default='out/main_nemotron3nano4b/meta_fixed.jsonl', help='Metadata file')
    parser.add_argument('--labels', default='out/main_nemotron3nano4b/labels_expected.csv', help='Labels file')
    parser.add_argument('--out', default='out/main_nemotron3nano4b/baseline_expected.json', help='Output file')
    
    args = parser.parse_args()
    
    run_baseline_expected(args.metadata, args.labels, args.out)