import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from typing import Collection, Sequence, Literal, Dict, Tuple, Callable, Optional
# from types import FrozenList
import utils
from abc import ABC, abstractmethod


DEFAULTS = utils.Defaults


class BaseAutoEncoder(nn.Module, ABC):
    """Abstract class for encoding multiple varying-frequency multi-variate 
    timeseries data"""
    def __init__(
            self, 
            window_sizes: Sequence[int],
            encoding_dim: int,
            num_transformer_layers: Sequence[int],
            dims: Sequence[int],
            activation_func: Callable,
            nheads: Sequence[int],
            device: torch.device,
            dropout: float=.1,
            layer_norm_eps: float=1e-4,
            dtype: torch.dtype=torch.float32):
        """
        :param window_sizes: number of periods in the inputs
        :param encoding_dim: number of dimensions in the encoded vector
            Different inputs will be encoded into the same-dimensional vector
            and concatenated
        :param num_transformer_layers: expressed a Sequence
        """
        super().__init__()
        self.window_sizes = window_sizes
        self.dims = Tuple(dims)
        self.num_transformer_layers = Tuple(num_transformer_layers)
        self.nheads = Tuple(nheads)
        self.linear_encoder_layers, self.linear_decoder_layers = [], []
        self.transformer_encoders, self.transformer_decoders = [], []
        self.num_inputs = len(num_transformer_layers)
        def create_encoder(encoder_layer: nn.TransformerEncoderLayer, num_layers: int):
            transformer_encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer, 
                    num_layers=num_layers,
                    )
            return transformer_encoder
        def create_decoder(decoder_layer: nn.TransformerDecoderLayer, num_layers: int):
            transformer_decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer, 
                num_layers=num_layers)
            return transformer_decoder
        def create_linear_encoder(dim: int, window_size: int):
            linear_encoder = nn.Sequential(
                nn.Flatten(1, -1),
                nn.BatchNorm1d(num_features=dim * window_size, device=device, dtype=dtype),
                nn.Linear(dim * window_size, dim * window_size // 4, device=device, dtype=dtype),
                nn.Linear(dim * window_size // 4, dim * window_size // 4 ** 2, device=device, dtype=dtype),
                nn.Linear(dim * window_size // 4 ** 2, dim * window_size // 4 ** 3, device=device, dtype=dtype),
                nn.Linear(dim * window_size // 4 ** 3, encoding_dim, device=device, dtype=dtype)
            )
            return linear_encoder
        def create_linear_decoder(dim: int, window_size: int):
            linear_decoder = nn.Sequential(
                nn.Linear(encoding_dim, dim * window_size // 4 ** 3, device=device, dtype=dtype),
                nn.Linear(dim * window_size // 4 ** 3, dim * window_size // 4 ** 2, device=device, dtype=dtype),
                nn.Linear(dim * window_size // 4 ** 2, dim * window_size // 4, device=device, dtype=dtype),
                nn.Linear(dim * window_size // 4, dim * window_size, device=device, dtype=dtype),
                nn.BatchNorm1d(num_features=dim * window_size, device=device, dtype=dtype),
                nn.Unflatten(-1, (self.window_size, self.dim)),
            )
            return linear_decoder
        self.tanh = nn.Tanh()
        self.linear_encoder = nn.Linear(self.num_inputs, 1)
        self.linear_decoder = nn.Linear(1, self.num_inputs)
        for i, (dim, nhead, num_transformer_layer, window_size) in enumerate(
            zip(
                dims, nheads, num_transformer_layers, window_sizes)
                ):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=nhead, batch_first=True, device=device, 
                dtype=dtype, dropout=dropout, layer_norm_eps=layer_norm_eps)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=dim, nhead=nhead, batch_first=True, 
                device=device, dtype=dtype, layer_norm_eps=layer_norm_eps)
            encoder_model = create_encoder(encoder_layer, num_transformer_layer)
            decoder_model = create_decoder(decoder_layer, num_transformer_layer)
            linear_decoder = create_linear_encoder(dim, window_size)
            linear_encoder = create_linear_decoder(dim, window_size)
            exec(f"""self.transformer_encoder_{i} = encoder_model
                 self.transformer_decoder_{i} = decoder_model
                 self.linear_encoder_{i} = linear_encoder
                 self.linear_decoder_{i} = linear_decoder
                 """)
            self.transformer_encoders.append(encoder_model)
            self.transformer_decoders.append(decoder_model)
            self.linear_encoder_layers.append(linear_encoder)
            self.linear_decoder_layers.append(linear_decoder)
        self.activation_func = activation_func

    def encode(
            self, 
            inputs: Sequence[Tuple[torch.tensor, Optional[torch.tensor]]],
            padding_masks: Sequence[torch.tensor]
            ) -> Tuple[torch.tensor, Tuple[torch.tensor]]:
        """encode the inputs i"""
        embeddings, memories = [], []
        for input, mask, transformer_encoder, linear_encoder in zip(
            inputs, padding_masks, self.transformer_encoders, self.linear_encoder_layers):
            x_ = transformer_encoder(input, src_key_padding_mask=mask)
            memories.append(x_)
            embedded = linear_encoder(x_)
            embeddings.append(embedded)
        _embedding = torch.concat(embeddings, dim=0)
        embedding = self.linear_encoder(_embedding)
        embedding = self.tanh(embedding)
        return (embedding, memories)
    
    def decode(
            self, 
            embedding: torch.tensor, 
            memories: Sequence[torch.tensor]) -> Sequence[torch.tensor]:
        _embeddings = self.linear_decoder(embedding)
        reconstructed_xs = []
        for i in range(self.num_inputs):
            _output = self.linear_decoder_layers[i](_embeddings(i))
            output = self.transformer_decoders[i](_output, memory=memories[i])
            reconstructed_xs.append(output)
        return output

    def forward(self, x, padding_mask: torch.tensor) -> torch.tensor:
        x_, z = self.encode(x, padding_mask=padding_mask)
        y_ = self.decode(z, x_)
        return y_
    
    def __call__(self, x, padding_mask) -> torch.tensor:
        return self.forward(x, padding_mask)

class MacroAutoEncoder(nn.Module):
    """Autoencoder on macroeconomic conditions"""
    def __init__(
            self, 
            window_size: int=DEFAULTS.window_size_weeks,
            encoding_dims: int=5,
            num_transformer_layers: int=10,
            dim: int=5,
            nhead: int=5):
        super().__init__()
        self.window_size = window_size
        self.dim = dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_transformer_layers)
        self.linear_encoder_ = nn.Sequential(
            # nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=3),
            nn.Flatten(1, -1),
            nn.BatchNorm1d(num_features=dim * window_size),
            nn.Linear(dim * window_size, dim * window_size // 4),
            nn.Linear(dim * window_size // 4, dim * window_size // 4 ** 2),
            nn.Linear(dim * window_size // 4 ** 2, dim * window_size // 4 ** 3),
            nn.Linear(dim * window_size // 4 ** 3, encoding_dims)
        )
        self.linear_decoder = nn.Sequential(
            nn.Linear(encoding_dims, dim * window_size // 4 ** 3),
            nn.Linear(dim * window_size // 4 ** 3, dim * window_size // 4 ** 2),
            nn.Linear(dim * window_size // 4 ** 2, dim * window_size // 4),
            nn.Linear(dim * window_size // 4, dim * window_size),
            nn.BatchNorm1d(num_features=dim * window_size),
            nn.Unflatten(-1, (self.window_size, self.dim)),
            # nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=num_transformer_layers)
        self.tanh = nn.Tanh()

    def encode(self, x: torch.tensor, padding_mask: torch.tensor) -> Tuple[torch.tensor]:
        x_ = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        # x_ = torch.flatten(x_, 1, 2)
        z = self.linear_encoder(x_)
        z = self.tanh(z)
        return x_, z
    
    def decode(self, z: torch.tensor, memory: torch.tensor) -> torch.tensor:
        y_ = self.linear_decoder(z)
        y = self.transformer_decoder(y_, memory=memory)
        return y

    def forward(self, x, padding_mask: torch.tensor) -> torch.tensor:
        x_, z = self.encode(x, padding_mask=padding_mask)
        y_ = self.decode(z, x_)
        return y_
    
    def __call__(self, x, padding_mask) -> torch.tensor:
        return self.forward(x, padding_mask)
    

class FundamentalsAutoEncoder(BaseAutoEncoder):
    def __init__(self):
        """        Initialize the object.

        Raises:
            NotImplementedError: This method is not implemented and should be overridden in a subclass.
        """

        raise NotImplementedError


class MacroAutoEncoder(BaseAutoEncoder):
    def __init__(self):
        """        Initialize the object.

        Raises:
            NotImplementedError: This method is not implemented and should be overridden in subclasses.
        """

        raise NotImplementedError

class PriceAutoEncoder(BaseAutoEncoder):
    def __init__(self):
        """        Initialize the object.

        Raises:
            NotImplementedError: This method is not implemented and should be overridden in a subclass.
        """

        raise NotImplementedError
    
class EstimatesAutoEncoder(BaseAutoEncoder):
    def __init__(self):
        """        Initialize the object.

        Raises:
            NotImplementedError: This method is not implemented and should be overridden in the derived classes.
        """

        raise NotImplementedError

class PricePredictionModel(BaseAutoEncoder):
    def __init__(self):
        """        Initialize the object.

        Raises:
            NotImplementedError: This method is not implemented and should be overridden in a subclass.
        """

        raise NotImplementedError