import os
import shutil
import sys

import click


def shell(command, exit_status=0):
    """
    Run command through shell and return exit status if exit status of command run match with given exit status.

    Args:
        command: (str) Command string which runs through system shell.
        exit_status: (int) Expected exit status of given command run.

    Returns: actual_exit_status

    """
    actual_exit_status = os.system(command)
    if actual_exit_status == exit_status:
        return 0
    return actual_exit_status


def validate_and_exit(expected_out_status=0, **kwargs):
    if all([arg == expected_out_status for arg in kwargs.values()]):
        # Expected status, OK
        sys.exit(0)
    else:
        # Failure
        print_console_centered("Summary Results")
        fail_count = 0
        for component, exit_status in kwargs.items():
            if exit_status != expected_out_status:
                click.secho(f"{component} failed.", fg="red")
                fail_count += 1

        print_console_centered(f"{len(kwargs) - fail_count} success, {fail_count} failure")
        click.secho("\nTo fix formatting issues:", fg="yellow")
        click.secho("1. Install development dependencies:", fg="cyan")
        click.secho('   pip install -e ."[dev]"', fg="green")
        click.secho("\n2. Run code formatting:", fg="cyan")
        click.secho("   python -m scripts.run_code_style format", fg="green")
        sys.exit(1)


def print_console_centered(text: str, fill_char="="):
    w, _ = shutil.get_terminal_size((80, 20))
    print(f" {text} ".center(w, fill_char))
