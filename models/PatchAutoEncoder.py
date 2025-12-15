import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------- FiLM Layer ----------------
class FiLMLayer(nn.Module):
    def __init__(self, feature_dim, cond_dim, hidden_dim=32):
        """
        FiLM layer applies feature-wise modulation:
          output = gamma * x + beta,
        where gamma and beta are generated from a conditioning vector.
        """
        super(FiLMLayer, self).__init__()
        self.gamma_fc = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        self.beta_fc = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def forward(self, x, cond):
        # x: (B, N, feature_dim)
        # cond: (B, N, cond_dim) or (B, cond_dim) which is broadcastable
        gamma = self.gamma_fc(cond)
        beta = self.beta_fc(cond)
        return gamma * x + beta

# ---------------- AdaPointTr Block (Transformer Block with Dropout) ----------------
class AdaPointTrBlock(nn.Module):
    def __init__(self, feature_dim, num_heads=4, dropout_prob=0.1):
        super(AdaPointTrBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.dropout_attn = nn.Dropout(dropout_prob)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        self.dropout_ffn = nn.Dropout(dropout_prob)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        attn_output = self.dropout_attn(attn_output)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout_ffn(ffn_output)
        x = self.norm2(x + ffn_output)
        return x

# ---------------- Patch Encoder with Transformer and Semantic Conditioning ----------------
class PatchEncoderTransformerWithSem(nn.Module):
    def __init__(self, 
                 input_dim=3,
                 encoder_blocks=3,
                 feature_dim=128,
                 latent_dim=64,
                 pos_enc_dim=16,
                 sem_dim=32,
                 num_classes=8,
                 dropout_prob=0.1):
        """
        Encoder for patch points that fuses positional information and semantic conditioning,
        and then uses transformer blocks to produce per-patch latent descriptors.
        """
        super(PatchEncoderTransformerWithSem, self).__init__()
        
        # Embedding layers.
        self.input_embed = nn.Linear(input_dim, feature_dim)
        self.pos_enc = nn.Linear(input_dim, pos_enc_dim)
        self.pos_proj = nn.Linear(pos_enc_dim, feature_dim)
        
        # Semantic embedding for patch-level semantic label.
        self.sem_embedding = nn.Embedding(num_classes, sem_dim)  
        
        # FiLM layer: condition on the semantic embedding.
        self.film = FiLMLayer(feature_dim, cond_dim=sem_dim)
        
        # Transformer encoder blocks.
        self.encoder_blocks = nn.ModuleList([
            AdaPointTrBlock(feature_dim, num_heads=4, dropout_prob=dropout_prob)
            for _ in range(encoder_blocks)
        ])
        
        # Spatial attention for pooling.
        self.spatial_att = nn.Sequential(
            nn.Linear(input_dim + pos_enc_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Global pooling and projection to latent space.
        self.latent_fc = nn.Linear(feature_dim, latent_dim)
    
    def forward(self, x, semantic_labels, mask=None):
        """
        Args:
            x: (B, N, 3) input patch points.
            semantic_labels: (B,) LongTensor containing the patch-level semantic label.
            mask: (B, N) Boolean mask.
        Returns:
            latent: (B, latent_dim) latent descriptor.
        """
        B, N, _ = x.shape
        # Initial per-point embedding.
        feat = self.input_embed(x)  # (B, N, feature_dim)
        pos_encoding = self.pos_enc(x)  # (B, N, pos_enc_dim)
        pos_proj = self.pos_proj(pos_encoding)  # (B, N, feature_dim)
        feat = feat + pos_proj  # fuse positional info
        
        # Obtain semantic embedding (per patch).
        sem_emb = self.sem_embedding(semantic_labels)  # (B, sem_dim)
        sem_emb_exp = sem_emb.unsqueeze(1).expand(B, N, -1)
        # Apply FiLM conditioning.
        feat = self.film(feat, sem_emb_exp)
        
        # Process through transformer blocks.
        for block in self.encoder_blocks:
            feat = block(feat)
        
        # Spatial attention pooling.
        att_input = torch.cat([x, pos_encoding], dim=-1)
        att_logits = self.spatial_att(att_input)
        if mask is not None:
            att_logits = att_logits.masked_fill(mask.unsqueeze(-1)==0, -1e9)
        att_weights = torch.softmax(att_logits, dim=1)
        
        pooled = torch.sum(feat * att_weights, dim=1)
        latent = self.latent_fc(pooled)
        return latent


# ---------------- Folding Upsampling Block ----------------
class FoldingUpsamplingBlock(nn.Module):
    def __init__(self, feature_dim, grid_size=4, output_dim=3):
        """
        Args:
            feature_dim (int): Dimension of the local point features.
            grid_size (int): Grid size along one dimension (total up_factor = grid_size^2).
            output_dim (int): Typically 3 for 3D point clouds.
        """
        super(FoldingUpsamplingBlock, self).__init__()
        self.grid_size = grid_size
        self.up_factor = grid_size * grid_size
        
        # The MLP takes as input the concatenated coarse features and 2D grid coordinates.
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, output_dim)
        )
    
    def forward(self, features, coords):
        """
        Args:
            features: (B, P, feature_dim) Coarse point features.
            coords: (B, P, 3) Coarse point coordinates.
        Returns:
            new_coords: (B, P * up_factor, 3) Upsampled point coordinates.
        """
        B, P, D = features.shape
        # Create a fixed 2D grid in the range [-1, 1].
        grid_range = torch.linspace(-1, 1, self.grid_size, device=features.device)
        grid_x, grid_y = torch.meshgrid(grid_range, grid_range, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)  # (up_factor, 2)
        grid = grid.unsqueeze(0).unsqueeze(0).expand(B, P, -1, -1)
        
        # Expand features for each coarse point.
        features_exp = features.unsqueeze(2).expand(B, P, self.up_factor, D)
        # Concatenate the grid coordinates with the features.
        mlp_input = torch.cat([features_exp, grid], dim=-1).reshape(B, P * self.up_factor, D + 2)
        
        # Use the MLP to predict offsets.
        offsets = self.mlp(mlp_input)  # (B, P * up_factor, 3)
        # Expand coarse coordinates and add predicted offsets.
        coords_exp = coords.unsqueeze(2).expand(B, P, self.up_factor, 3).reshape(B, P * self.up_factor, 3)
        new_coords = coords_exp + offsets
        return new_coords

# ---------------- Patch Decoder with Learned Coarse Offsets ----------------
class PatchDecoderFolding(nn.Module):
    def __init__(self, 
                 latent_dim=64, 
                 coarse_points=256, 
                 feature_dim=128,
                 upsample_factors=4,  # Optional, overrides grid_size.
                 decoder_hidden_dims=[128, 256],
                 output_dim=3,
                 norm_factor=100.0  # Normalization factor for the bbox extents.
                 ):
        """
        Args:
            latent_dim (int): Dimension of the latent vector.
            coarse_points (int): Number of coarse points.
            feature_dim (int): Dimension of the features for each coarse point.
            upsample_factors (int): Upsampling factor (grid_size^2).
            decoder_hidden_dims (list): Hidden layer sizes for decoding the latent vector.
            output_dim (int): Typically 3 for 3D point clouds.
            norm_factor (float): Normalization factor for bbox extents.
        """
        super(PatchDecoderFolding, self).__init__()
        grid_size = int(math.sqrt(upsample_factors))
        self.coarse_points = coarse_points
        self.feature_dim = feature_dim
        self.norm_factor = norm_factor

        # Map latent vector to coarse point features.
        self.coarse_fc = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dims[0], coarse_points * feature_dim)
        )

        # Predict a coarse point confidence (mask) for each coarse point.
        self.coarse_mask_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )

        # New head: Predict learned offsets to adjust the coarse points.
        self.coarse_offset_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 3)
        )

        # Folding upsampling block.
        self.folding_block = FoldingUpsamplingBlock(feature_dim, grid_size=grid_size, output_dim=output_dim)

        # Final mask prediction for upsampled points.
        self.final_mask_mlp = nn.Sequential(
            nn.Linear(feature_dim + 1, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )

    def sample_in_bbox(self, extent, orientation):
        """
        Samples coarse points uniformly within a bounding box defined by the normalized extents,
        then rotates them using the provided orientation matrices.
        
        Args:
            extent: (B, 3) tensor with the side lengths of the bbox.
            orientation: (B, 3, 3) tensor with rotation matrices.
        Returns:
            points: (B, coarse_points, 3) uniformly sampled points in local space.
        """
        B = extent.shape[0]
        eps = 1e-6
        # Normalize the extents.
        norm_extent = extent / self.norm_factor
        # Sample uniformly in a box defined by normalized extents: [-norm_extent/2, norm_extent/2].
        points = torch.rand(B, self.coarse_points, 3, device=extent.device) * (norm_extent.unsqueeze(1) + eps) - norm_extent.unsqueeze(1) / 2
        # Rotate the points using the orientation matrices.
        points = torch.bmm(points, orientation.transpose(1, 2))
        return points

    def forward(self, latent, extent, orientation):
        """
        Args:
            latent: (B, latent_dim) latent descriptors.
            extent: (B, 3) raw extents of the bounding boxes.
            orientation: (B, 3, 3) orientation matrices.
        Returns:
            coarse_local: (B, coarse_points, 3) learned coarse point coordinates in local space.
            coarse_mask: (B, coarse_points, 1) coarse point confidence mask.
            final_mask: (B, coarse_points * up_factor, 1) mask for upsampled points.
            final_local: (B, coarse_points * up_factor, 3) upsampled point coordinates in local space.
        """
        B = latent.shape[0]
        # Generate per-coarse-point features.
        coarse_feat = self.coarse_fc(latent).view(B, self.coarse_points, -1)
        # Predict coarse point confidence.
        coarse_mask = self.coarse_mask_head(coarse_feat)

        # Sample initial coarse points uniformly in the bbox (using extent and orientation).
        coarse_init = self.sample_in_bbox(extent, orientation)
        # Predict learned offsets for each coarse point.
        coarse_offsets = self.coarse_offset_head(coarse_feat)
        # Update coarse point positions.
        coarse_local = coarse_init + coarse_offsets

        # Upsample using the folding block.
        final_local = self.folding_block(coarse_feat, coarse_local)

        # Prepare inputs for the final mask prediction.
        up_factor = self.folding_block.up_factor
        coarse_feat_rep = coarse_feat.repeat_interleave(up_factor, dim=1)
        coarse_mask_rep = coarse_mask.repeat_interleave(up_factor, dim=1)
        final_mask_input = torch.cat([coarse_feat_rep, coarse_mask_rep], dim=-1)
        final_mask = self.final_mask_mlp(final_mask_input)

        return coarse_local, coarse_mask, final_mask, final_local
    

# ---------------- Combined Autoencoder Wrapper ----------------
class PatchAutoencoderTransformer(nn.Module):
    def __init__(self, encoder: PatchEncoderTransformerWithSem, decoder: nn.Module):
        super(PatchAutoencoderTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x, semantic_labels, mask=None, center=None, extent=None, orientation=None):
        """
        Args:
            x: (B, N, 3) input patch points.
            semantic_labels: (B,) LongTensor for patch-level semantic label.
            mask: (B, N) Boolean mask indicating valid points.
        Returns:
            latent: (B, latent_dim) latent descriptor.
            coarse_global: (B, coarse_points, 3) coarse point cloud in global coordinates.
            reconstruction: (B, N_out, 3) fine reconstruction in global coordinates.
        """
        center = center.unsqueeze(1)  
        
        # Convert to local coordinates.
        x_local = x - center
        
        # Encode using the local coordinates.
        latent = self.encoder(x_local, semantic_labels, mask)
        # Decode to get coarse and fine reconstructions in local coordinates.
        coarse_local, coarse_mask, reconstruction_mask, reconstruction_local = self.decoder(latent, extent, orientation)
        # Convert back to global coordinates.
        coarse_global = coarse_local + center
        reconstruction = reconstruction_local + center
        
        return latent, coarse_global, coarse_mask, reconstruction_mask, reconstruction


