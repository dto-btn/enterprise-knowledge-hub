"""
Torch implementations for embedding backend.  safetensors files
"""
import os
from typing import List, Any
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from provider.embedding.base import EmbeddingBackendProvider


class TorchEmbeddingBackendProvider(EmbeddingBackendProvider):
    """
    Torch implementations for embedding backend.  safetensors files
    """
    def __init__(self, model_name: str, logger, device: str = "cuda", max_seq_len: int = 512):
        
        tokenizer: Any
        model: Any
        if device == "cpu":
            local_dir = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF/snapshots/370f27d7550e0def9b39c1f16d3fbaa13aa67728/")
            tokenizer = AutoTokenizer.from_pretrained(local_dir,
                                                           gguf_file="Qwen3-Embedding-0.6B-Q8_0.gguf", local_files_only=True)
            model = AutoModel.from_pretrained(local_dir, gguf_file="Qwen3-Embedding-0.6B-Q8_0.gguf", local_files_only=True).to(device)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)

        model.eval()
        super().__init__(model_name, device, max_seq_len, tokenizer, model, logger)

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
