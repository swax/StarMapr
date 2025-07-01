"""
Colored print utilities for StarMapr
Provides consistent colored output across all scripts
"""

# ANSI color codes
class Colors:
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_error(message):
    """Print error message in red"""
    print(f"{Colors.RED}{message}{Colors.RESET}")

def print_summary(message):
    """Print summary message in blue"""
    print(f"{Colors.BLUE}{message}{Colors.RESET}")