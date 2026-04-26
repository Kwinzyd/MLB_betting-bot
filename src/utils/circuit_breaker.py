import time
from functools import wraps
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class CircuitBreakerOpenException(Exception):
    """Raised when the circuit breaker is open and fast-failing requests."""
    pass

class CircuitBreaker:
    """
    A simple circuit breaker to prevent hammering an unresponsive API.
    
    - CLOSED: Normal operation.
    - OPEN: Fast-fails requests until recovery_timeout has elapsed.
    - HALF-OPEN: Allows one test request through to check if the service recovered.
    """
    def __init__(
        self, 
        failure_threshold: int = 5, 
        recovery_timeout: float = 60.0,
        exceptions: tuple = (Exception,)
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.exceptions = exceptions
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    logger.info(f"Circuit breaker HALF-OPEN for '{func.__name__}'. Testing connection...")
                    self.state = "HALF-OPEN"
                else:
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker OPEN. Fast-failing request for '{func.__name__}'."
                    )
            
            try:
                result = func(*args, **kwargs)
            except self.exceptions as e:
                self._record_failure(func.__name__, e)
                raise
            
            self._record_success(func.__name__)
            return result
        
        return wrapper

    def _record_failure(self, func_name: str, exception: Exception):
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.state == "HALF-OPEN" or self.failures >= self.failure_threshold:
            if self.state != "OPEN":
                logger.error(
                    f"Circuit breaker TRIPPED for '{func_name}' after {self.failures} failures. "
                    f"Suppressing calls for {self.recovery_timeout}s. Last error: {exception}"
                )
            self.state = "OPEN"

    def _record_success(self, func_name: str):
        if self.state == "HALF-OPEN":
            logger.info(f"Circuit breaker CLOSED for '{func_name}'. Service restored.")
        self.failures = 0
        self.state = "CLOSED"


class AsyncCircuitBreaker:
    """An async-compatible circuit breaker."""
    def __init__(
        self, 
        failure_threshold: int = 5, 
        recovery_timeout: float = 60.0,
        exceptions: tuple = (Exception,)
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.exceptions = exceptions
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"

    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    logger.info(f"Circuit breaker HALF-OPEN for '{func.__name__}'. Testing connection...")
                    self.state = "HALF-OPEN"
                else:
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker OPEN. Fast-failing request for '{func.__name__}'."
                    )
            
            try:
                result = await func(*args, **kwargs)
            except self.exceptions as e:
                self._record_failure(func.__name__, e)
                raise
            
            self._record_success(func.__name__)
            return result
        
        return wrapper

    def _record_failure(self, func_name: str, exception: Exception):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.state == "HALF-OPEN" or self.failures >= self.failure_threshold:
            if self.state != "OPEN":
                logger.error(
                    f"Circuit breaker TRIPPED for '{func_name}' after {self.failures} failures. "
                    f"Suppressing calls for {self.recovery_timeout}s. Last error: {exception}"
                )
            self.state = "OPEN"

    def _record_success(self, func_name: str):
        if self.state == "HALF-OPEN":
            logger.info(f"Circuit breaker CLOSED for '{func_name}'. Service restored.")
        self.failures = 0
        self.state = "CLOSED"