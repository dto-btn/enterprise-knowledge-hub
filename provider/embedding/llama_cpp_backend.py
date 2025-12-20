"""
Llama_CPP implementations for embedding backend.  GGUF files
"""
import os
import gc
from dataclasses import dataclass
from typing import List, Optional, Callable
import numpy as np
from transformers import AutoTokenizer, AutoModel
from llama_cpp import Llama
from provider.embedding.base import EmbeddingBackendProvider


@dataclass
class LlamaCppEmbedder:
    """
    data class to hold values for llama embedder
    """
    #tweak these values depending on model.
    # Eventually expose this so we can mod on the fly.
    model_path: str
    n_ctx:int = 2048
    n_gpu_layers:int = -1 # -1 = as many as possible
    n_threads:Optional[int] = None
    n_batch: int = 512
    n_ubatch: int = 512
    pooling_type: int = 0
    llm: Llama

    def __post__init__(self):
        self.llm = Llama(
            model_path=self.model_path,
            embedding=True,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            n_threads=self.n_threads,
            n_batch=self.n_batch,
            n_ubatch=self.n_ubatch,
            pooling_type=self.pooling_type,
        )

class LlamaCPPEmbeddingBackendProvider(EmbeddingBackendProvider):
    """
    Llama_CPP implementations for embedding backend.  GGUF files
    """
    llama_embedder: LlamaCppEmbedder
    local_dir = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF/snapshots/370f27d7550e0def9b39c1f16d3fbaa13aa67728/")
    def __init__(self, model_name: str, logger, device: str = "cuda", max_seq_len: int = 512):
        self.model_name = model_name
        self.device = device
        self.max_seq_len = max_seq_len
        self.llama_embedder = LlamaCppEmbedder(self.local_dir)
        #for testing purposes here
        # local_dir = os.path.expanduser(
            # "~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF/
            # snapshots/370f27d7550e0def9b39c1f16d3fbaa13aa67728/")
        # self.tokenizer = AutoTokenizer.from_pretrained(local_dir,
        # gguf_file="Qwen3-Embedding-0.6B-Q8_0.gguf", local_files_only=True)
        # self.model = AutoModel.from_pretrained(local_dir, gguf_file="Qwen3-Embedding-0.6B-Q8_0.gguf",
        # local_files_only=True).to(device)

        if device == "cpu":
            self.tokenizer = self.local_token()
            self.model = self.local_model()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

        self.model.eval()
        super().__init__(model_name, device, max_seq_len, self.tokenizer, self.model, logger)

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

    def embed(self, text: List[str], normalize: bool = True, **kwargs) -> np.ndarray:
        """
        Returns shape: (B,D) float32
        """
        emb = self.llama_embedder.llm.embed(text, normalize=normalize, truncate=True)

        return np.asarray(emb, dtype=np.float32)

    def detect_max_batch_size(self,
                              start_n_batch: int = 128,
                              max_n_batch_cap: int = 8192,
                              prob_texts_per_call: int = 8,
                              ubatch_policy: Callable[[int], int] = lambda nb: min(nb, 512),
                              normalize: bool = True,
                              warmup: bool = True,
                              **kwargs):
        probe_text = "This is a dummy sentence for batch size testing."
        probe_inputs = [probe_text] * prob_texts_per_call

        def try_run(n_batch: int) -> bool:
            n_ubatch = ubatch_policy(n_batch)

            try:
                llm = Llama(
                    model_path=self.local_dir,  #this needs to change after testing
                    embedding=True,              # required :contentReference[oaicite:6]{index=6}
                    n_ctx=self.llama_embedder.n_ctx,
                    n_gpu_layers=self.llama_embedder.n_gpu_layers,
                    n_threads=self.llama_embedder.n_threads,
                    n_batch=n_batch,
                    n_ubatch=n_ubatch,
                    pooling_type=self.llama_embedder.pooling_type,
                    verbose=False,
                )

                if warmup:
                    _ = llm.embed([probe_inputs[0]], normalize=normalize, truncate=True)

                _ = llm.embed(probe_inputs, normalize=normalize, truncate=True)
                return True

            except Exception :
                print(f"[FAIL] n_batch={n_batch}, n_ubatch={n_ubatch}")
                return False

            finally:
                # try to free native + python memory between probes
                try:
                    del llm
                except UnboundLocalError:
                    pass
                gc.collect()

        lo = start_n_batch
        hi = lo
        while True:
            next_hi = hi * 2
            if next_hi > max_n_batch_cap:
                next_hi = max_n_batch_cap

            if next_hi == hi:
                break

            if try_run(next_hi):
                lo = next_hi
                hi = next_hi
                if hi >= max_n_batch_cap:
                    break
            else:
                hi = next_hi
                break

        # If we never found a failure, lo is our cap
        if lo == max_n_batch_cap:
            return lo, ubatch_policy(lo)

        # Binary search between lo (pass) and hi (fail) to find max pass
        left = lo
        right = hi
        best = lo

        while left + 1 < right:
            mid = (left + right) // 2
            if try_run(mid):
                best = mid
                left = mid
            else:
                right = mid

        return best, ubatch_policy(best)

    def batched(self, iterable, token_budget: int, token_counter, max_items: int | None = None, **kwargs):
        """
        token_counter(text) -> int  (number of tokens for that text)
        """
        batch = []
        tokens = 0

        for text in iterable:
            t = token_counter(text)

            # if single item exceeds budget, still yield it alone (or truncate upstream)
            if batch and (tokens + t > token_budget or (max_items and len(batch) >= max_items)):
                yield batch
                batch = []
                tokens = 0

            batch.append(text)
            tokens += t

        if batch:
            yield batch
