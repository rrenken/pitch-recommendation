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
        
        # Projecting continuous featueres to specified output_dim//2 (less expressive)
        self.continuous_proj = nn.Linear(continuous_dim, output_dim // 2)

        # Projecting continuous featueres to larger dimension space (more expressive)
        #self.continuous_proj = nn.Linear(continuous_dim, output_dim * 3 // 4)
        
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

