from typing import Callable, Generic, List, TypeVar
from dataclasses import dataclass

T = TypeVar("T")

@dataclass
class BatchHandler(Generic[T]):
    def __init__(self, process_batch: Callable[[List[T]], None], batch_size: int):
        self.process_batch = process_batch
        self.batch_size = batch_size
        self.item_list: List[T] = []

    def __call__(self, item: T) -> None:
        self.item_list.append(item)

        # don't process until we have the required batch_size
        if len(self.item_list) < self.batch_size:
            return

        batch = self.item_list
        self.item_list = []  

        self.process_batch(batch)
