import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer model"""
    def __init__(self, d_model, max_seq_length=50):
        super().__init__()
        
        # Create positional encodings
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Args:
            x: Tensor shape [batch_size, seq_len, embedding_dim]
        """
        # Add positional encoding
        return x + self.pe[:, :x.size(1)]


class PitchTransformer(nn.Module):
    """Decoder-only transformer for pitch sequence prediction"""
    def __init__(self, 
                 embed_dim=128, 
                 num_heads=8,
                 num_layers=6,
                 pitch_vocab_size=20,
                 dropout=0.1,
                 max_seq_len=20):
        """
        Args:
            embed_dim: Dimensionality of pitch embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            pitch_vocab_size: Number of unique pitch types to predict
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)
        
        # Transformer layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim*4,
            dropout=dropout,
            batch_first=True  # Important: our shape is [batch_size, seq_len, embed_dim]
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )
        
        # Final prediction head for pitch type
        self.pitch_classifier = nn.Linear(embed_dim, pitch_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, mask=None):
        """
        Args:
            src: Tensor shape [batch_size, seq_len, embed_dim]
            mask: Attention mask to avoid looking at future positions
                  Shape [batch_size, seq_len]
                  
        Returns:
            pitch_logits: Prediction logits for each pitch type
                         Shape [batch_size, seq_len, pitch_vocab_size]
        """
        # Add positional encoding
        x = self.positional_encoding(src)
        x = self.dropout(x)
        
        # Create attention mask for decoder-only (causal) transformer
        # This ensures predictions for position i can only use positions 0 to i-1
        seq_len = src.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=src.device) * float('-inf'),
            diagonal=1
        )
        
        # Create padding mask from attention mask if provided
        key_padding_mask = None
        if mask is not None:
            # Convert boolean mask (True = keep, False = mask) to the opposite
            key_padding_mask = ~mask
            
        # Pass through transformer decoder
        # For decoder-only, we use src as both src and tgt
        output = self.transformer_decoder(
            tgt=x,
            memory=x,  # In decoder-only, memory is unused but required by API
            tgt_mask=causal_mask,
            tgt_key_padding_mask=key_padding_mask
        )
        
        # Predict pitch type
        pitch_logits = self.pitch_classifier(output)
        
        return pitch_logits
    






    
    
    def generate(self, sequence, max_new_tokens=5):
        """
        Generate pitch predictions autoregressively.
        
        Args:
            sequence: Starting sequence of pitch embeddings
                     Shape [1, seq_len, embed_dim]
            max_new_tokens: Maximum number of new pitches to generate
            
        Returns:
            Complete sequence with generated pitches
            Shape [1, seq_len + max_new_tokens, embed_dim]
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get prediction for next token
                logits = self.forward(sequence)
                
                # Get the most likely next pitch (last position)
                next_pitch_logits = logits[:, -1, :]
                next_pitch_probs = F.softmax(next_pitch_logits, dim=-1)
                
                # Sample or take argmax
                next_pitch_idx = torch.argmax(next_pitch_probs, dim=-1)
                
                # This would be where you convert the pitch idx back to an embedding
                # For now we'll just use a placeholder
                next_pitch_embedding = torch.zeros_like(sequence[:, 0:1, :])
                
                # Append to sequence
                sequence = torch.cat([sequence, next_pitch_embedding], dim=1)
                
        return sequence