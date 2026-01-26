"""Service to facilitate graceful shutdown"""
import asyncio
from dataclasses import dataclass

@dataclass
class ShutdownService:
    """Service to facilitate graceful shutdown"""
    stop_event: asyncio.Event
    deadline: float = 30.0

    def __init__(self, deadline: float = 30.0):
        self.stop_event = asyncio.Event()
        self.deadline = deadline

    def request_stop(self):
        """Request for process to stop"""
        self.stop_event.set()

    def should_stop(self) -> bool:
        """Return true if and only if the internal flag is true."""
        return self.stop_event.is_set()
