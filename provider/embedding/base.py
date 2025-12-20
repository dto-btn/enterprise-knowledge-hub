"""
Embedding Provider base
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Any
import numpy as np

@dataclass
class EmbeddingBackendProvider(ABC):
    """
    Embedding Provider base
    """
    model_name: str
    device: str
    max_seq_length: int
    tokenizer: Any
    model: Any
    logger: logging.Logger

    @abstractmethod
    def set_device(self, device: str):
        """
        Set device type
        """
        raise NotImplementedError

    @abstractmethod
    def embed(self, text: List[str], **kwargs) -> np.ndarray:
        """
        embedding abstract method
        """
        raise NotImplementedError

    @abstractmethod
    def batched(self, **kwargs):
        """
        Docstring for batched

        :param self: Description
        :param kwargs: Description
        """
        raise NotImplementedError

    @abstractmethod
    def detect_max_batch_size(self, **kwargs):
        """
        detects max batch size
        """
        raise NotImplementedError
