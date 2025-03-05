import torch
import torch.nn as nn

class PitchEmbedding(nn.Module):
    def __init__(self, continuous_dim, categorical_info, output_dim):
        """
        Args:
            continuous_dim (int): Number of continuous features.
            categorical_info (dict): A dictionary where keys are categorical feature names and values are 
                                     dicts with keys 'num_categories' and 'embedding_dim'.
            output_dim (int): Final dimension for each pitch token embedding.
        """
        super(PitchEmbedding, self).__init__()
        self.continuous_dim = continuous_dim
        self.categorical_info = categorical_info
        
        # For continuous variables, project them into a fixed dimension.
        # Here, we project continuous features to half of the final output dimension.
        self.continuous_proj = nn.Linear(continuous_dim, output_dim // 2)
        
        # For categorical variables, create an embedding layer for each feature.
        self.categorical_embeddings = nn.ModuleDict()
        total_cat_dim = 0
        for key, info in categorical_info.items():
            self.categorical_embeddings[key] = nn.Embedding(num_embeddings=info['num_categories'],
                                                              embedding_dim=info['embedding_dim'])
            total_cat_dim += info['embedding_dim']
        
        # Combine continuous and categorical embeddings.
        # The concatenated vector has dimension (output_dim//2 + total_cat_dim).
        # We then project it to the final output_dim.
        self.final_proj = nn.Linear((output_dim // 2) + total_cat_dim, output_dim)
        
    def forward(self, continuous_inputs, categorical_inputs):
        """
        Args:
            continuous_inputs (Tensor): Shape (batch_size, continuous_dim)
            categorical_inputs (dict): Dictionary where keys correspond to feature names and
                                       values are tensors of shape (batch_size,)
        Returns:
            Tensor: Final pitch token embeddings of shape (batch_size, output_dim)
        """
        # Process continuous features
        cont_embed = self.continuous_proj(continuous_inputs)  # Shape: (batch_size, output_dim//2)
        
        # Process each categorical feature and collect their embeddings
        cat_embeds = []
        for key, embed_layer in self.categorical_embeddings.items():
            # Assume categorical_inputs[key] is a tensor of indices with shape (batch_size,)
            cat_embeds.append(embed_layer(categorical_inputs[key]))
        
        # Concatenate all categorical embeddings along the last dimension
        cat_embed = torch.cat(cat_embeds, dim=-1)  # Shape: (batch_size, total_cat_dim)
        
        # Concatenate continuous and categorical embeddings
        combined = torch.cat([cont_embed, cat_embed], dim=-1)
        # Project to final output dimension
        out = self.final_proj(combined)
        return out

# ------------------------------
# Example usage:
# Suppose you have 100 continuous fields and 13 categorical fields.
# For demonstration, here we use 4 categorical features as an example.

continuous_dim = 100

# Dictionary for categorical variables:
# Keys are the feature names, and for each you specify the number of unique categories and desired embedding dimension.
categorical_info = {
    'pitcher_id': {'num_categories': 500, 'embedding_dim': 32},
    'pitch_type': {'num_categories': 10, 'embedding_dim': 16},
    'batter_side': {'num_categories': 3, 'embedding_dim': 8},
    'game_situation': {'num_categories': 20, 'embedding_dim': 16},
    # ... add the rest as needed to reach a total of 13 categorical features.
}

output_dim = 256  # Final token embedding dimension

# Instantiate the embedding model
pitch_embedding_model = PitchEmbedding(continuous_dim, categorical_info, output_dim)

# Create dummy data for a batch of 32 pitches
batch_size = 32
continuous_inputs = torch.randn(batch_size, continuous_dim)

categorical_inputs = {
    'pitcher_id': torch.randint(0, 500, (batch_size,)),
    'pitch_type': torch.randint(0, 10, (batch_size,)),
    'batter_side': torch.randint(0, 3, (batch_size,)),
    'game_situation': torch.randint(0, 20, (batch_size,))
    # ... ensure you provide inputs for all categorical features defined in categorical_info.
}

# Generate the embeddings
embeddings = pitch_embedding_model(continuous_inputs, categorical_inputs)
print("Embedding shape:", embeddings.shape)  # Expected output: (32, 256)
