import sys


class PrayaasException(Exception):
    """Wraps an error with the file and line where it was raised."""

    def __init__(self, error: Exception):
        _, _, tb = sys.exc_info()
        location = "unknown"
        if tb is not None:
            location = f"{tb.tb_frame.f_code.co_filename}:{tb.tb_lineno}"
        super().__init__(f"{error} (at {location})")
