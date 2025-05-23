import sys
import time
import threading
from colorama import init, Fore, Style

# Initialize colorama
init()

def print_loading(message):
    """Print loading animation with color"""
    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    i = 0
    while not hasattr(print_loading, "stop"):
        sys.stdout.write(f"\r{Fore.CYAN}{message} {chars[i]}{Style.RESET_ALL}")
        sys.stdout.flush()
        time.sleep(0.1)
        i = (i + 1) % len(chars)
    sys.stdout.write("\r" + " " * (len(message) + 2) + "\r")
    sys.stdout.flush()

def start_loading(message):
    """Start loading animation in a separate thread"""
    print_loading.stop = False
    thread = threading.Thread(target=print_loading, args=(message,))
    thread.daemon = True
    thread.start()
    return thread

def stop_loading(thread):
    """Stop loading animation"""
    print_loading.stop = True
    thread.join()

def print_success(message):
    """Print success message in green"""
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")

def print_warning(message):
    """Print warning message in yellow"""
    print(f"{Fore.YELLOW}⚠ {message}{Style.RESET_ALL}")

def print_error(message):
    """Print error message in red"""
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")

def print_info(message):
    """Print info message in blue"""
    print(f"{Fore.BLUE}ℹ {message}{Style.RESET_ALL}")

def show_loading_animation(message, duration=2, color=Fore.CYAN):
    """Show a loading animation with dots and spinner"""
    loading_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    dots = "⠄⠆⠇⠋⠙⠸⠴⠦⠧⠇⠏"
    i = 0
    j = 0
    start_time = time.time()
    while time.time() - start_time < duration:
        sys.stdout.write(f"\r{color}{message} {dots[j]} {loading_chars[i]}{Style.RESET_ALL}")
        sys.stdout.flush()
        time.sleep(0.1)
        i = (i + 1) % len(loading_chars)
        j = (j + 1) % len(dots)
    sys.stdout.write("\r" + " " * 50 + "\r")  # Clear the line
    sys.stdout.flush() 