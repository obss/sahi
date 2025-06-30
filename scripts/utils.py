import shutil


def print_console_centered(text: str, fill_char="="):
    """Print text centered in console with fill characters."""
    w, _ = shutil.get_terminal_size((80, 20))
    print(f" {text} ".center(w, fill_char))
