import os

def debug_print(*args, **kwargs):
    if os.getenv("DEBUG") == "1":
        print(*args, **kwargs)
