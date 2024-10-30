from abc import ABC, abstractmethod
from typing import Any, List


class OfflineRequester(ABC):
    def __init__(self):
        self._request_queue: List[Any] = []

    @abstractmethod
    def add_request(self, item: Any) -> bool:
        raise NotImplementedError("Subclasses must implement add_request()")

    @abstractmethod
    def submit_request(self) -> bool:
        raise NotImplementedError("Subclasses must implement submit_request()")

    def clear_queue(self) -> None:
        self._request_queue.clear()

    def get_queue_size(self) -> int:
        return len(self._request_queue)


