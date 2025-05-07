import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import time
import os
from transformer import PitchTransformer

class PitchTransformerTrainer:
    """Trainer class for the PitchTransformer model"""
    def __init__(self, 
                model, 
                learning_rate=5e-5, 
                weight_decay=0.01,
                device=None,
                class_weights=None):
        """
        Args:
            model: PitchTransformer model
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            device: Device to use for training (cpu or cuda)
            class_weights: Optional 1D tensor of shape [num_classes] for class-weighted loss
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        # ignore_index=-100 means we'll use -100 to mask out padded positions
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device), ignore_index=-100)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # History for plotting
        self.train_losses = []
        self.val_losses = []
        
    def prepare_data(self, embeddings_path, batch_size=32):
        """
        Load and prepare data for training.
        
        Args:
            embeddings_path: Path to saved embeddings file
            batch_size: Batch size for training
            
        Returns:
            train_loader, val_loader: DataLoader objects for training and validation
        """
        print(f"Loading data from {embeddings_path}")
        data = torch.load(embeddings_path)
        
        # Extract data
        pitch_embeddings = data['embeddings']  # [num_atbats, max_seq_len, embed_dim]
        attention_mask = data['attention_mask']  # [num_atbats, max_seq_len]
        
        # We need pitch type labels for prediction task
        # If you don't have them in your saved file, we need to extract them
        if 'pitch_types' not in data:
            print("Warning: No pitch type labels found in saved data. Please add them.")
            print("For now, creating dummy pitch types for demonstration.")
            pitch_types = torch.randint(0, 10, (pitch_embeddings.shape[0], pitch_embeddings.shape[1]))
        else:
            pitch_types = data['pitch_types']
        
        # Create inputs and targets for next pitch prediction
        # Target is the next pitch in sequence (shifted by 1)
        inputs = []
        targets = []
        masks = []
        
        print("Preparing training examples...")
        
        # Find max sequence length for all valid sequences
        max_seq_len = 0
        for i in range(len(pitch_embeddings)):
            seq_len = attention_mask[i].sum().item()
            if seq_len > 1:  # Only consider valid sequences
                max_seq_len = max(max_seq_len, seq_len)
                
        print(f"Maximum sequence length: {max_seq_len}")
        
        # Now process each sequence with proper padding
        for i in range(len(pitch_embeddings)):
            seq_len = attention_mask[i].sum().item()
            
            if seq_len <= 1:  # Skip sequences with just one pitch or less
                continue
                
            # Input: pad to max length
            padded_input = torch.zeros(max_seq_len, pitch_embeddings.shape[2])
            padded_input[:seq_len] = pitch_embeddings[i, :seq_len]
            inputs.append(padded_input)
            
            # Target: pitch types shifted by 1 (predict next pitch)
            # We pad with -100 which will be ignored in loss
            target = torch.full((max_seq_len,), -100, dtype=torch.long)
            target[:seq_len-1] = pitch_types[i, 1:seq_len]
            targets.append(target)
            
            # Mask for valid positions
            mask = torch.zeros(max_seq_len, dtype=torch.bool)
            mask[:seq_len] = True
            masks.append(mask)





        
        if not inputs:
            raise ValueError("No valid sequences found. Check your data.")
            
        # Stack into tensors
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)
        masks = torch.stack(masks)
        
        # Create dataset
        dataset = TensorDataset(inputs, targets, masks)
        
        # Split into train/val
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size, 
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size, 
            shuffle=False
        )
        
        print(f"Created {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training", leave=False):
            inputs, targets, masks = [b.to(self.device) for b in batch]
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs, masks)
            
            # Compute loss
            # Reshape for cross entropy: [batch_size*seq_len, num_classes]
            loss = self.criterion(
                outputs.view(-1, outputs.size(-1)), 
                targets.view(-1)
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                inputs, targets, masks = [b.to(self.device) for b in batch]
                
                # Forward pass
                outputs = self.model(inputs, masks)
                
                # Compute loss
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)), 
                    targets.view(-1)
                )
                
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, num_epochs=10, early_stopping_patience=10):
        """Full training loop with early stopping and epoch timing"""
        print(f"Training on {self.device}")
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_epoch = -1
        best_model_path = None

        for epoch in range(num_epochs):
            epoch_start = time.time()
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            epoch_end = time.time()
            epoch_duration = epoch_end - epoch_start
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train loss: {train_loss:.4f}, "
                  f"Val loss: {val_loss:.4f}, "
                  f"Duration: {epoch_duration:.2f} seconds")

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_epoch = epoch + 1
                best_model_path = f"../models/pitch_transformer_best.pt"
                self.save_checkpoint(best_model_path)
            else:
                epochs_no_improve += 1

            # Save checkpoint every 25 epochs or last epoch
            if (epoch + 1) % 25 == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(f"../models/pitch_transformer_epoch_{epoch+1}.pt")
                self.plot_losses()

            # Early stopping
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs. Best val loss: {best_val_loss:.4f} at epoch {best_epoch}.")
                print(f"Best model saved to {best_model_path}")
                break

        return self.train_losses, self.val_losses
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
        print(f"Checkpoint saved to {path}")
        
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f"Checkpoint loaded from {path}")
        
    def plot_losses(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('../figures/training_loss.png')
        plt.close()

    

    # Add this method to your PitchTransformerTrainer class

    def evaluate_accuracy(self, data_loader):
        """
        Calculate accuracy metrics on a dataset
        
        Returns:
            dict: Dictionary with accuracy metrics
        """
        self.model.eval()
        correct_top1 = 0
        correct_top3 = 0
        total_valid = 0
        
        with torch.no_grad():
            for batch in data_loader:
                inputs, targets, masks = [b.to(self.device) for b in batch]
                
                # Forward pass
                outputs = self.model(inputs, masks)
                
                # Get top-k predictions
                probs = torch.softmax(outputs, dim=-1)
                top1_preds = torch.argmax(probs, dim=-1)
                top3_preds = torch.topk(probs, k=3, dim=-1).indices
                
                # Compare with targets
                valid_positions = (targets != -100)
                targets_flat = targets[valid_positions]
                top1_flat = top1_preds[valid_positions]
                
                # Count correct predictions
                correct_top1 += (top1_flat == targets_flat).sum().item()
                
                # For top-3, check if target is in top 3 predictions
                for i, row in enumerate(top3_preds):
                    for j, pred_row in enumerate(row):
                        if not valid_positions[i, j]:
                            continue
                        if targets[i, j].item() in pred_row:
                            correct_top3 += 1
                        total_valid += 1
        
        # Calculate accuracy
        top1_acc = correct_top1 / total_valid if total_valid > 0 else 0
        top3_acc = correct_top3 / total_valid if total_valid > 0 else 0
        
        return {
            'top1_accuracy': top1_acc,
            'top3_accuracy': top3_acc,
            'total_valid': total_valid
        }

if __name__ == "__main__":
    import os
    from transformer import PitchTransformer

    start_time = time.time()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directories if needed
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../figures', exist_ok=True)

    # Path to embeddings (update as needed)
    embeddings_path = '../data/embeddings/pitch_embeddings_with_types_test.pt'

    # Load pitch type map to get vocab size
    embedding_data = torch.load(embeddings_path)
    if 'pitch_type_map' in embedding_data:
        num_pitch_types = len(embedding_data['pitch_type_map'])
    elif 'pitch_types' in embedding_data:
        num_pitch_types = int(embedding_data['pitch_types'].max().item()) + 1
    else:
        raise ValueError("pitch_type_map or pitch_types not found in embedding data.")

    # Compute class weights for imbalanced pitch types
    pitch_types = embedding_data['pitch_types']
    valid_targets = pitch_types[pitch_types != -100].flatten()
    num_classes = num_pitch_types  # Ensure this matches model output
    counts = torch.bincount(valid_targets, minlength=num_classes)
    class_weights = 1.0 / (counts.float() + 1e-6)  # Avoid division by zero
    class_weights = class_weights / class_weights.sum() * num_classes  # Normalize

    # Model hyperparameters (match your embedding/sequence setup)
    model = PitchTransformer(
        embed_dim=embedding_data['embeddings'].shape[2],
        num_heads=8,
        num_layers=6,
        pitch_vocab_size=num_pitch_types,
        dropout=0.1,
        max_seq_len=embedding_data['embeddings'].shape[1]
    )

    # Trainer
    trainer = PitchTransformerTrainer(
        model=model,
        learning_rate=5e-5,
        weight_decay=0.01,
        device=device,
        class_weights=class_weights
    )

    # Prepare data
    train_loader, val_loader = trainer.prepare_data(
        embeddings_path=embeddings_path,
        batch_size=32
    )

    # Train
    train_losses, val_losses = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        early_stopping_patience=10
    )

    # Optionally evaluate accuracy
    acc = trainer.evaluate_accuracy(val_loader)
    print(f"Validation Top-1 Accuracy: {acc['top1_accuracy']:.3f}, Top-3 Accuracy: {acc['top3_accuracy']:.3f}")

    end_time = time.time()
    duration = end_time - start_time
    print(f"Pretraining complete! Total duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")