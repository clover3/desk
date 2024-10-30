import time
from collections import deque


class RateLimiter:
    def __init__(self, max_requests=600, time_window=600):
        """
        Initialize rate limiter with specified constraints

        Args:
            max_requests (int): Maximum number of requests allowed in the time window
            time_window (int): Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()  # Store timestamps of requests

    def acquire(self):
        """
        Check if a new request can be made and wait if necessary.
        Returns when it's safe to make a new request.
        """
        current_time = time.time()

        # Remove timestamps older than the time window
        while self.requests and current_time - self.requests[0] >= self.time_window:
            self.requests.popleft()

        # If we're at the limit, calculate wait time
        if len(self.requests) >= self.max_requests:
            # Calculate how long to wait for the oldest request to expire
            wait_time = self.requests[0] + self.time_window - current_time
            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                # After waiting, clean up old timestamps again
                current_time = time.time()
                while self.requests and current_time - self.requests[0] >= self.time_window:
                    self.requests.popleft()

        # Add current request timestamp
        self.requests.append(current_time)

    def get_current_usage(self):
        """
        Get current rate limit usage statistics

        Returns:
            dict: Contains current usage information
        """
        current_time = time.time()

        # Clean up old timestamps
        while self.requests and current_time - self.requests[0] >= self.time_window:
            self.requests.popleft()

        return {
            'current_requests': len(self.requests),
            'max_requests': self.max_requests,
            'time_window': self.time_window,
            'remaining_requests': self.max_requests - len(self.requests)
        }