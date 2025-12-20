"""
Torch implementations for embedding backend.  safetensors files
"""
import os
from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from provider.embedding.base import EmbeddingBackendProvider

class TorchEmbeddingBackendProvider(EmbeddingBackendProvider):
    """
    Torch implementations for embedding backend.  safetensors files
    """
    local_dir = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF/snapshots/370f27d7550e0def9b39c1f16d3fbaa13aa67728/")
    def __init__(self, model_name: str, logger, device: str = "cuda", max_seq_len: int = 512):
        if device == "cpu":
            self.tokenizer = self.local_token()
            self.model = self.local_model()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

        self.model.eval()
        self.model.to(self.device)
        super().__init__(model_name, device, max_seq_len, self.tokenizer, self.model, logger)

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
            max_length=self.max_seq_length,
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

    def local_token(self):
        """
        Docstring for local_token

        :param self: Description
        """
        tokenizer = AutoTokenizer.from_pretrained(self.local_dir, gguf_file="Qwen3-Embedding-0.6B-Q8_0.gguf",
                                                local_files_only=True)
        return tokenizer

    def local_model(self):
        """
        Docstring for local_model

        :param self: Description
        """
        model = AutoModel.from_pretrained(self.local_dir, gguf_file="Qwen3-Embedding-0.6B-Q8_0.gguf",local_files_only=True)
        return model

    def detect_max_batch_size(self,
        start_batch: int = 1,
        max_batch_cap: int = 4096,
    ) -> int:
        """
        detect max batch size for torch
        """
        print("================detectbatch")

        # if default and device is cpu, then lower batch cap
        if max_batch_cap == 4096 and self.device == 'cpu':
            max_batch_cap = 1024

        dummy_text = "This is a dummy sentence for batch size testing."

        def can_run(batch_size: int) -> bool:
            try:
                texts = [dummy_text] * batch_size
                inputs = self.tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                ).to(self.device)

                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device=self.device)
                with torch.no_grad():
                    _ = self.model(**inputs)
                return True
            except RuntimeError as e:
                print ('runtime======' + str(e).lower())
                if "out of memory" in str(e).lower():

                    # Clear OOM state
                    print(f"OOM at batch_size={batch_size}")
                    # torch.cuda.empty_cache()
                    return False
                return True

        # Phase 1: exponential search to find an upper bound where OOM happens
        low = 0
        high = start_batch

        while high <= max_batch_cap:
            print(f"Trying batch_size={high}...")
            if can_run(high):
                low = high
                high *= 2
            else:
                break

        if low == 0:
            raise RuntimeError("Even batch_size=1 does not fit in GPU memory.")

        # If we never hit OOM up to max_batch_cap, just return that cap
        if high > max_batch_cap:
            print(f"Reached max_batch_cap={max_batch_cap} without OOM.")
            return max_batch_cap

        # Phase 2: binary search between low (good) and high (bad)
        print(f"Binary search between {low} (ok) and {high} (OOM)")
        while low + 1 < high:
            mid = (low + high) // 2
            print(f"Trying batch_size={mid}...")
            if can_run(mid):
                low = mid
            else:
                high = mid

        print(f"Max safe batch size = {low}")
        return low

    def batched(self, iterable, batch_size: int):
        """
        Write meaningful stuff
        """
        batch = []
        for item in iterable:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
