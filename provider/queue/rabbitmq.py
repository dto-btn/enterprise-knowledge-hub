from collections.abc import Iterator
from contextlib import contextmanager
import pika
from pika.adapters.blocking_connection import BlockingChannel

from provider.queue.base import QueueProvider

class RabbitMQProvider(QueueProvider):
    """RabbitMQ queue configuration provider"""

    @contextmanager
    def channel(self) -> Iterator[BlockingChannel]:
        connection = pika.BlockingConnection(pika.ConnectionParameters(url=self.url))
        try:
            channel = connection.channel()
            yield channel
        finally:
            channel.close()
            connection.close()


    @contextmanager
    def read(self, queue_name: str) -> dict[str, object]:
        """Read from the specified RabbitMQ queue."""
        with self.channel() as channel:
            method_frame = channel.queue_declare(queue=queue_name, durable=True)
            if method_frame != pika.spec.Queue.DeclareOk:
                raise ValueError(f"Queue {queue_name} does not exist.")
            method_frame, header_frame, body = channel.basic_get(queue=queue_name, auto_ack=True)
            if method_frame:
                yield {"method_frame": method_frame, "header_frame": header_frame, "body": body}

    @contextmanager
    def write(self, queue_name: str, message: dict[str, object]) -> None:
        """Write to the specified RabbitMQ queue."""
        # Implementation to write to RabbitMQ queue