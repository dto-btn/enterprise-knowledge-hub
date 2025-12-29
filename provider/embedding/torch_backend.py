"""
Torch implementations for embedding backend.  safetensors files
"""
from typing import List
import numpy as np
import torch
from provider.embedding.base import EmbeddingBackendProvider


class TorchEmbeddingBackend(EmbeddingBackendProvider):
    """
    Torch implementations for embedding backend.  safetensors files
    """
    def __init__(self, model_name: str, device: str = "cuda", max_seq_len: int = 512):
        super().__init__(model_name=model_name, device=device, max_seq_len=max_seq_len)

    def set_device(self, device: str):
        """
        set device type
        """
        self.device = device

    def embed(self, text: List[str]) -> np.ndarray:
        """
        embed implementation for torch
        """
        #this means i'm retokenizing i think?  doing it twice.
        #chunk token by text, tokenizes, then decodes back to text.
            # maybe don't decode.
        # check how long it takes, we can deal with this when we split embed
        # and tokenizing to ensure optimizing by time
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad(): # pylint: disable=no-member
            outputs = self.model(**inputs)

        # simple mean pooling
        # change this to be dynamic based on modeL?
        last_hidden = outputs.last_hidden_state      # [B, T, D]
        mask = inputs["attention_mask"].unsqueeze(-1)  # [B, T, 1]
        masked = last_hidden * mask
        summed = masked.sum(dim=1)
        lengths = mask.sum(dim=1)                    # [B, 1]
        embeddings = summed / lengths

        return embeddings.cpu().numpy()
