# Credit to the CS-231n course at Stanford, from which this assignment is adapted
import numpy as np
import copy
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        ##|Q.1.a|##########################################################################
        # TODO: Initialize the following layers and parameters to perform attention
        # This class assumes that the input dimension for query, key and value is embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        ###################################################################################

    def forward(self, query, key, value, attn_mask=None):
        N, S, D = query.shape
        N, T, D = value.shape
        assert key.shape == value.shape
       
        ##|Q.1.a|##########################################################################
        # TODO: Compute attention 
        # Project query, key and value
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        # Compute dot-product attention
        dot_product = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(D)

        if attn_mask is not None:
            # Convert att_mask which is multiplicative, to an additive mask
            # Hint: If mask[i,j] = 0, we want softmax(QKT[i,j] + additive_mask[i,j]) to be 0
            # Think about what inputs make softmax 0.
            additive_mask = (1 - attn_mask) * -1e12
            dot_product += additive_mask
        
        # Apply softmax, dropout, and use value
        y = torch.bmm(self.dropout(F.softmax(dot_product, dim=-1)), value)
        ###################################################################################
        return y  


class MultiHeadAttentionLayer(AttentionLayer):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__(embed_dim, dropout)
        self.num_heads = num_heads

        ##|Q.1.b|##########################################################################
        # TODO: Initialize the following layers and parameters to perform attention
        self.head_proj = nn.Linear(embed_dim, embed_dim)
        ###################################################################################

    def forward(self, query, key, value, attn_mask=None):
        H = self.num_heads
        N, S, D = query.shape
        N, T, D = value.shape
        assert key.shape == value.shape

        ##|Q.1.b|##########################################################################
        # TODO: Compute multi-head attention
        # Project query, key and value, and split the embedding across num_heads after projection
        D_h = D // H
        query = self.query_proj(query).view(N, S, H, D_h).transpose(1, 2)  # (N, H, S, D_h)
        key = self.key_proj(key).view(N, T, H, D_h).transpose(1, 2)  # (N, H, T, D_h)
        value = self.value_proj(value).view(N, T, H, D_h).transpose(1, 2)  # (N, H, T, D_h)

        # Compute dot-product attention separately for each head. Don't forget the scaling value!
        dot_product = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(D_h)  # (N, H, S, T)

        if attn_mask is not None:
            # Convert att_mask which is multiplicative, to an additive mask
            # Hint: If mask[i,j] = 0, we want softmax(QKT[i,j] + additive_mask[i,j]) to be 0
            # Think about what inputs make softmax 0.
            additive_mask = (1 - attn_mask) * -1e12
            dot_product += additive_mask
        
        # Apply softmax, dropout, and use value
        y = torch.matmul(self.dropout(F.softmax(dot_product, dim=-1)), value)

        # Concat embeddings from different heads, and project
        output = self.head_proj(y.transpose(1, 2).contiguous().view(N, S, D))
        ###################################################################################
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        ##|Q.1.c|##########################################################################
        # TODO: Use torch.nn.Embedding to create the encoding. Initialize dropout layer.
        self.encoding = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        ###################################################################################
      
    def forward(self, x):
        N, S, D = x.shape
        ##|Q.1.c|##########################################################################
        # TODO: Add the encoding to x
        output = x + self.encoding(torch.arange(S, device=x.device).unsqueeze(0))
        output = self.dropout(output)
        ###################################################################################
        return output


class SelfAttentionBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super().__init__()
        ##|Q.1.d|##########################################################################
        # TODO: Initialize the following. Use MultiHeadAttentionLayer for self_attn.
        self.self_attn = MultiHeadAttentionLayer(input_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(input_dim)
        ###################################################################################
       
    def forward(self, seq, mask):
        ##|Q.1.d|##########################################################################
        # TODO: Self-attention on the sequence, using the mask. Add dropout to attention layer output.
        # Then add a residual connection to the original input, and finally apply normalization.
        out = self.self_attn(seq, seq, seq, attn_mask=mask)
        out = self.layernorm(seq + self.dropout(out))
        ###################################################################################
        return out


class CrossAttentionBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super().__init__()
        ##|Q.1.d|##########################################################################
        # TODO: Initialize the following. Use MultiHeadAttentionLayer for cross_attn.
        self.cross_attn = MultiHeadAttentionLayer(input_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)
        ###################################################################################
       
    def forward(self, seq, cond):
        ##|Q.1.d|##########################################################################
        # TODO: Cross-attention on the sequence, using conditioning. Add dropout to attention layer output.
        # Then add a residual connection to the original input, and finally apply normalization.
        out = self.cross_attn(seq, cond, cond)
        out = self.norm(seq + self.dropout(out))
        ###################################################################################
        return out

class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super().__init__()
        ##|Q.1.d|##########################################################################
        # TODO: Initialize the following. 
        # MLP has the following layers: linear, relu, dropout, linear; hidden dim of linear is given by dim_feedforward
        assert activation in ['relu', 'swiglu']
        self.activation = activation
        
        self.fc_1 = nn.Linear(input_dim, dim_feedforward if self.activation == 'relu' else 2 * dim_feedforward)
        self.fc_2 = nn.Linear(dim_feedforward, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)
        ###################################################################################

    def forward(self, seq):
        ##|Q.1.d|##########################################################################
        # TODO: MLP on the sequence. Add dropout to mlp layer output.
        # Then add a residual connection to the original input, and finally apply normalization.
        x = self.fc_1(seq)
        if self.activation == 'relu':
            x = F.relu(x)
        else:
            u, v = x.chunk(2, dim=-1)
            x = F.silu(u) * v
        x = self.dropout(x)
        x = self.fc_2(x)
        out = self.norm(seq + self.dropout(x))
        ###################################################################################
        return out


class DecoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super().__init__()
        self.self_atn_block = SelfAttentionBlock(input_dim, num_heads, dropout)
        self.cross_atn_block = CrossAttentionBlock(input_dim, num_heads, dropout)
        self.feedforward_block = FeedForwardBlock(input_dim, num_heads, dim_feedforward, dropout, activation)

    def forward(self, seq, cond, mask):
        out = self.self_atn_block(seq, mask)
        out = self.cross_atn_block(out, cond)
        return self.feedforward_block(out)


class TransformerDecoder(nn.Module):
    def __init__(self, word_to_idx, idx_to_word, input_dim, embed_dim, num_heads=4, num_layers=2, max_length=50, device='cuda', activation='relu'):
        """
        Construct a new TransformerDecoder instance.
        Inputs:
            - word_to_idx: A dictionary giving the vocabulary. It contains V entries.
            and maps each string to a unique integer in the range [0, V).
            - input_dim: Dimension of input image feature vectors.
            - embed_dim: Embedding dimension of the transformer.
            - num_heads: Number of attention heads.
            - num_layers: Number of transformer layers.
            - max_length: Max possible sequence length.
        """
        super().__init__()

        vocab_size = len(word_to_idx)
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self.idx_to_word = idx_to_word
        
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads, activation=activation) for _ in range(num_layers)])
        
        self.caption_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self._null)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=max_length)
        self.feature_embedding = nn.Linear(input_dim, embed_dim)
        self.score_projection = nn.Linear(embed_dim, vocab_size) 

        self.apply(self._init_weights)
        self.device = device 
        self.to(device)

    def get_data_embeddings(self, features, captions):
        ##|Q.1.e|##########################################################################
        # TODO - get caption and feature embeddings 
        # Don't forget position embeddings for captions!
        # expected caption embedding output shape : (N, T, D)
        caption_embedding = self.positional_encoding(self.caption_embedding(captions))

        # Unsqueeze feature embedding along dimension 1
        # expected feature embedding output shape : (N, 1, D)
        feature_embedding = self.feature_embedding(features).unsqueeze(1)
        ###################################################################################
        return feature_embedding, caption_embedding

    def get_causal_mask(self, _len):
        ##|Q.1.e|##########################################################################
        # TODO: Get causal mask. This should be a matrix of shape (_len, _len). 
        # This mask is multiplicative.
        # Setting mask[i,j] = 0 means jth element of the sequence is not used to predict the ith element of the sequence.
        mask = torch.tril(torch.ones(_len, _len, dtype=torch.bool)).to(self.device)
        ###################################################################################
        return mask
                                      
    def forward(self, features, captions):
        """
        Given image features and caption tokens, return a distribution over the
        possible tokens for each timestep. Note that since the entire sequence
        of captions is provided all at once, we mask out future timesteps.
        Inputs:
         - features: image features, of shape (N, D)
         - captions: ground truth captions, of shape (N, T)
        Returns:
         - scores: score for each token at each timestep, of shape (N, T, V)
        """
        features_embed, captions_embed = self.get_data_embeddings(features, captions)
        mask = self.get_causal_mask(captions_embed.shape[1])
        mask = mask.to(captions_embed.dtype)
        
        output = captions_embed
        for layer in self.layers:
            output = layer(output, features_embed, mask=mask)

        scores = self.score_projection(output)
        return scores

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def sample(self, features, max_length=30):
        """
        Given image features, use greedy decoding to predict the image caption.
        Inputs:
         - features: image features, of shape (N, D)
         - max_length: maximum possible caption length
        Returns:
         - captions: captions for each example, of shape (N, max_length)
        """
        with torch.no_grad():
            features = torch.Tensor(features).to(self.device)
            N = features.shape[0]

            # Create an empty captions tensor (where all tokens are NULL).
            captions = self._null * np.ones((N, max_length), dtype=np.int32)

            # Create a partial caption, with only the start token.
            partial_caption = self._start * np.ones(N, dtype=np.int32)
            partial_caption = torch.LongTensor(partial_caption).to(self.device)
            # [N] -> [N, 1]
            partial_caption = partial_caption.unsqueeze(1)

            for t in range(max_length):

                # Predict the next token (ignoring all other time steps).
                output_logits = self.forward(features, partial_caption)
                output_logits = output_logits[:, -1, :]

                # Choose the most likely word ID from the vocabulary.
                # [N, V] -> [N]
                word = torch.argmax(output_logits, axis=1)

                # Update our overall caption and our current partial caption.
                captions[:, t] = word.cpu().numpy()
                word = word.unsqueeze(1)
                partial_caption = torch.cat([partial_caption, word], dim=1)

            return captions
