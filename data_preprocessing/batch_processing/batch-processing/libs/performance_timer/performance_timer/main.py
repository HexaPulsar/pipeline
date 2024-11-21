import contextlib
import time

class PerformanceTimer:
    """
    Context manager for measuring execution time of code blocks.
    """

    def __init__(self, title: str = ""):
        self.start_time = None
        self.title = title

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            end_time = time.perf_counter()
            elapsed_time = end_time - self.start_time
            print(f"""
            🤔~~~~
                  Elapsed time for '{self.title}': {elapsed_time:.4f} seconds
            🤔~~~~
            """)
