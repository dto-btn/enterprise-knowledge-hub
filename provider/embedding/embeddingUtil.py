import gc
import json
import time
from typing import Dict, Iterable, Tuple, Callable
from llama_cpp import Llama
from transformers import PreTrainedTokenizerBase, AutoTokenizer, AutoModel
import os
import torch
import math



class EmbeddingUtil:
  
    @staticmethod
    def chunk_text_by_tokens(
            text: str,
            tokenizer: PreTrainedTokenizerBase,
            max_tokens: int = 512,
            overlap_tokens: int = 64,
        ):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            chunks = []
            start = 0

            while start < len(tokens):
                end = start + max_tokens
                chunk_tokens = tokens[start:end]
                chunk_text = tokenizer.decode(chunk_tokens)
                chunks.append(chunk_text)

                if end >= len(tokens):
                    break

                # move forward but keep an overlap
                start = end - overlap_tokens

            return chunks

    #make this abstract?  the yield is wiki specific
    @staticmethod
    def article_to_chunks(article: Dict, tokenizer, max_tokens=512, overlap_tokens=64): 
        chunks = EmbeddingUtil.chunk_text_by_tokens(article["xml_content"], tokenizer, max_tokens, overlap_tokens)
        for i, chunk_text in enumerate(chunks):
            yield {
                "id": f'{article["page_id"]}-{i}',
                "text": chunk_text,
                "metadata": {
                    "article_id": article["page_id"],
                    "title": article.get("title"),
                    "chunk_index": i,
                },
            }
                  
    @staticmethod
    def batched (iterable: Iterable[Dict], batch_size: int):
        batch = []
        for item in iterable:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
          
    @staticmethod
    def detect_max_batch_size_torch(
        # model_name: str,
        max_seq_len: int = 512,
        device: str = "cuda",
        start_batch: int = 1,
        max_batch_cap: int = 4096,
    ):
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # model = AutoModel.from_pretrained(model_name)
        
        # if default and device is cpu, then lower batch cap
        if max_batch_cap == 4096 and device == 'cpu':
            max_batch_cap: 1024
        
        #test
        local_dir = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF/snapshots/370f27d7550e0def9b39c1f16d3fbaa13aa67728/")
        tokenizer = AutoTokenizer.from_pretrained(local_dir, gguf_file="Qwen3-Embedding-0.6B-Q8_0.gguf", local_files_only=True)
        model = AutoModel.from_pretrained(local_dir, gguf_file="Qwen3-Embedding-0.6B-Q8_0.gguf", local_files_only=True)
        
        model.to(device)
        model.eval()

        dummy_text = "This is a dummy sentence for batch size testing."

        def can_run(batch_size: int) -> bool:
            try:
                texts = [dummy_text] * batch_size
                inputs = tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=max_seq_len,
                    return_tensors="pt",
                ).to(device)

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device=device)
                with torch.no_grad():
                    _ = model(**inputs)

                #test logic
                # if batch_size < 10: 
                #     return True
                # else:
                #     raise RuntimeError("out of memory")
            except RuntimeError as e:
                print ('runtime======' + str(e).lower())
                if "out of memory" in str(e).lower():
                    
                    # Clear OOM state
                    print(f"OOM at batch_size={batch_size}")
                    # torch.cuda.empty_cache()
                    return False
                else:
                    raise

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
    
    def detect_max_batch_size_llamacpp(
        model_path: str,
        n_ctx: int,
        n_gpu_layers: int,
        n_threads: int | None = None,
        pooling_type: int = 0,
        # probe controls:
        start_n_batch: int = 128,
        max_n_batch_cap: int = 8192,
        # microbatch: keep <= n_batch; often 256-1024 is a reasonable range to test
        ubatch_policy: Callable[[int], int] = lambda nb: min(nb, 512),
        # workload controls:
        probe_texts_per_call: int = 8,
        probe_text_chars: int = 6000,
        normalize: bool = True,
        warmup: bool = True,
        verbose: bool = True,
    ) -> Tuple[int, int]:
        """
        Returns (best_n_batch, best_n_ubatch)

        This measures the maximum *token batch parameter* (n_batch) that works,
        not "how many texts".
        """

        dummy_text = "This is a dummy sentence for batch size testing."
        probe_inputs = [dummy_text] * probe_texts_per_call

        def try_run(n_batch: int) -> bool:
            n_ubatch = ubatch_policy(n_batch)

            try:
                llm = Llama(
                    model_path=model_path,
                    embedding=True,              # required :contentReference[oaicite:6]{index=6}
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    n_threads=n_threads,
                    n_batch=n_batch,
                    n_ubatch=n_ubatch,
                    pooling_type=pooling_type,
                    verbose=False,
                )

                if warmup:
                    _ = llm.embed([probe_inputs[0]], normalize=normalize, truncate=True)

                _ = llm.embed(probe_inputs, normalize=normalize, truncate=True)
                return True

            except Exception as e:
                if verbose:
                    print(f"[FAIL] n_batch={n_batch}, n_ubatch={n_ubatch} -> {type(e).__name__}: {e}")
                return False

            finally:
                # try to free native + python memory between probes
                try:
                    del llm
                except UnboundLocalError:
                    pass
                gc.collect()
                time.sleep(0.05)

        # 1) Exponential search for upper bound
        lo = start_n_batch
        if not try_run(lo):
            # If even the start fails, return something safe-ish
            return 0, 0

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

        # 2) Binary search between lo (pass) and hi (fail) to find max pass
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