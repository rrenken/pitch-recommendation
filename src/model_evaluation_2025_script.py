# model_evaluation_2025_script.py
# Script to evaluate the pretrained model on 2025 data from the terminal
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path for imports
sys.path.append(os.path.abspath('..'))
from src.transformer import PitchTransformer
from src.pretraining import PitchTransformerTrainer

def evaluate_pitch_predictions(model, data_loader, pitch_type_map, device):
    model.eval()
    correct_top1 = 0
    correct_top3 = 0
    total_valid = 0
    all_preds = []
    all_targets = []
    pitch_type_correct = defaultdict(int)
    pitch_type_total = defaultdict(int)
    rev_map = {idx: pitch for pitch, idx in pitch_type_map.items()}
    with torch.no_grad():
        for batch_idx, (inputs, targets, masks) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            outputs = model(inputs, masks)
            probs = torch.softmax(outputs, dim=-1)
            top1_preds = torch.argmax(probs, dim=-1)
            top3_preds = torch.topk(probs, k=3, dim=-1).indices
            for i in range(len(inputs)):
                for j in range(inputs.size(1)-1):
                    if j+1 < inputs.size(1) and masks[i, j+1] and targets[i, j+1] != -100 and targets[i, j+1] != -1:
                        target_idx = targets[i, j+1].item()
                        pred_idx = top1_preds[i, j].item()
                        if pred_idx == target_idx:
                            correct_top1 += 1
                        if target_idx in top3_preds[i, j]:
                            correct_top3 += 1
                        if target_idx in rev_map:
                            pitch_type = rev_map[target_idx]
                            pitch_type_total[pitch_type] += 1
                            if pred_idx == target_idx:
                                pitch_type_correct[pitch_type] += 1
                        all_preds.append(pred_idx)
                        all_targets.append(target_idx)
                        total_valid += 1
    results = {
        'top1_accuracy': correct_top1 / total_valid if total_valid > 0 else 0,
        'top3_accuracy': correct_top3 / total_valid if total_valid > 0 else 0,
        'total_valid': total_valid,
        'pitch_type_accuracy': {pt: pitch_type_correct[pt] / pitch_type_total[pt] for pt in pitch_type_total if pitch_type_total[pt] > 0},
        'pitch_type_counts': dict(pitch_type_total),
        'all_preds': all_preds,
        'all_targets': all_targets
    }
    return results

def main():
    import time
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Load the pretrained model
    model_dir = '../models/'
    model_files = [f for f in os.listdir(model_dir) if f.startswith('pitch_transformer_epoch_') and f.endswith('.pt')]
    if not model_files:
        raise FileNotFoundError("No model checkpoints found! Please run pretraining first.")
    latest_model = sorted(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
    model_path = os.path.join(model_dir, latest_model)
    print(f"Using model checkpoint: {latest_model}")
    pretraining_data_path = '../data/embeddings/pitch_embeddings_with_types_test.pt'
    pretraining_data = torch.load(pretraining_data_path)
    pitch_type_map = pretraining_data['pitch_type_map']
    num_pitch_types = len(pitch_type_map)
    model = PitchTransformer(
        embed_dim=128,
        num_heads=8,
        num_layers=6,
        pitch_vocab_size=num_pitch_types,
        dropout=0.1,
        max_seq_len=20
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    # Load the 2025 data embeddings
    data_2025_path = '../data/embeddings/pitch_embeddings_2025_parquet.pt'
    data_2025 = torch.load(data_2025_path)
    embeddings_2025 = data_2025['embeddings']
    attention_mask_2025 = data_2025['attention_mask']
    print(f"Loaded embeddings with shape: {embeddings_2025.shape}")
    # Load the original 2025 data to extract pitch types
    df_2025 = pd.read_csv('../data/processed/pitcher_final_2025.csv')
    rev_pitch_map = {idx: pitch for pitch, idx in pitch_type_map.items()}
    unique_at_bats = df_2025['at_bat_id'].unique()
    pitch_types_2025 = []
    for at_bat_id in unique_at_bats:
        atbat_df = df_2025[df_2025['at_bat_id'] == at_bat_id]
        atbat_pitch_types = []
        for pt in atbat_df['pitch_type']:
            if pt in pitch_type_map:
                atbat_pitch_types.append(pitch_type_map[pt])
            else:
                atbat_pitch_types.append(-1)
        max_len = embeddings_2025.shape[1]
        padded_types = atbat_pitch_types + [-100] * (max_len - len(atbat_pitch_types))
        pitch_types_2025.append(padded_types)
    pitch_types_2025 = torch.tensor(pitch_types_2025)
    print(f"Extracted pitch types tensor with shape: {pitch_types_2025.shape}")
    eval_dataset = TensorDataset(embeddings_2025, pitch_types_2025, attention_mask_2025)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    print("Data preparation complete!")
    # Evaluate model
    eval_results = evaluate_pitch_predictions(model, eval_loader, pitch_type_map, device)
    print(f"\nEvaluation Results on 2025 Data:")
    print(f"Total valid prediction points: {eval_results['total_valid']:,}")
    print(f"Top-1 Accuracy: {eval_results['top1_accuracy']:.4f} ({eval_results['top1_accuracy']*100:.2f}%)")
    print(f"Top-3 Accuracy: {eval_results['top3_accuracy']:.4f} ({eval_results['top3_accuracy']*100:.2f}%)")
    # Save results as figures
    plt.figure(figsize=(10, 6))
    accuracies = [eval_results['top1_accuracy'], eval_results['top3_accuracy']]
    labels = ['Top-1', 'Top-3']
    plt.bar(labels, accuracies, color=['#3498db', '#2ecc71'])
    plt.ylim(0, 1.0)
    plt.title('2025 Pitch Prediction Accuracy', fontsize=16)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f"{v*100:.1f}%", ha='center', fontweight='bold')
    plt.savefig('../figures/2025_prediction_accuracy.png')
    plt.close()
    pitch_types = list(eval_results['pitch_type_accuracy'].keys())
    accuracies = list(eval_results['pitch_type_accuracy'].values())
    counts = [eval_results['pitch_type_counts'][pt] for pt in pitch_types]
    sorted_indices = np.argsort(accuracies)[::-1]
    pitch_types = [pitch_types[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]
    plt.figure(figsize=(14, 8))
    bars = plt.bar(pitch_types, accuracies, color='#3498db')
    plt.title('Prediction Accuracy by Pitch Type (2025)', fontsize=16)
    plt.xlabel('Pitch Type', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0, max(accuracies) + 0.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    for i, (bar, acc, count) in enumerate(zip(bars, accuracies, counts)):
        plt.text(bar.get_x() + bar.get_width()/2, acc + 0.01, f"{acc*100:.1f}%\n(n={count})", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig('../figures/2025_pitch_type_accuracy.png')
    plt.close()
    # Confusion matrix
    common_pitches = [pt for pt, count in eval_results['pitch_type_counts'].items() if count > 50]
    rev_map = {idx: pitch for pitch, idx in pitch_type_map.items()}
    common_idx = [i for i, t in enumerate(eval_results['all_targets']) if t in [pitch_type_map[pt] for pt in common_pitches]]
    filtered_preds = [eval_results['all_preds'][i] for i in common_idx]
    filtered_targets = [eval_results['all_targets'][i] for i in common_idx]
    labels = [rev_map.get(pitch_type_map[pt], 'UNK') for pt in common_pitches]
    label_indices = [pitch_type_map[pt] for pt in common_pitches]
    cm = confusion_matrix(filtered_targets, filtered_preds, labels=label_indices)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Pitch Prediction Confusion Matrix (2025)', fontsize=16)
    plt.xlabel('Predicted Pitch', fontsize=14)
    plt.ylabel('Actual Pitch', fontsize=14)
    plt.tight_layout()
    plt.savefig('../figures/2025_confusion_matrix.png')
    plt.close()
    print("Evaluation complete. Figures saved.")
    end_time = time.time()
    duration = end_time - start_time
    print(f"Total script duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")

if __name__ == "__main__":
    main()
