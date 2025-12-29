"""
Embedding Provider base
"""
from abc import ABC, abstractmethod
from typing import List
import numpy as np

from transformers import AutoTokenizer, AutoModel


class EmbeddingBackendProvider(ABC):
    """
    Embedding Provider base
    """
    model_name: str
    device: str
    max_seq_len: int
    tokenizer: object | None
    model: object | None

    def __init__(self, model_name: str, device: str = "cuda", max_seq_len: int = 512, load_model: bool = True):
        """Initialize common backend attributes and optionally load the HF model/tokenizer.

        Child classes should call `super().__init__(...)` to reuse this logic.
        """
        self.model_name = model_name
        self.device = device
        self.max_seq_len = max_seq_len
        self.tokenizer = None
        self.model = None

        if load_model:
            # Default shared model/tokenizer loading used by most backends
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            try:
                # Some model objects expose eval() to set inference mode
                self.model.eval()
            except Exception:
                pass

    def set_device(self, device: str):
        """
        Set device type
        """
        self.device = device

    @abstractmethod
    def embed(self, text: List[str]) -> np.ndarray:
        """
        embedding abstract method
        """
        raise NotImplementedError
