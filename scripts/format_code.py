#!/usr/bin/env python
"""Simple code formatting script for SAHI."""

import subprocess
import sys


def run_command(cmd):
    """Run a command and return exit code."""
    result = subprocess.run(cmd, shell=True)
    return result.returncode


def check_formatting():
    """Check code formatting without making changes."""
    print("Checking code formatting...")

    # Check linting
    lint_status = run_command("ruff check .")

    # Check formatting
    format_status = run_command("ruff format --check .")

    if lint_status == 0 and format_status == 0:
        print("\n✅ All checks passed!")
        return 0
    else:
        print("\n❌ Formatting issues found. Run 'python scripts/format_code.py fix' to fix them.")
        return 1


def fix_formatting():
    """Fix code formatting issues."""
    print("Fixing code formatting...")

    # Fix linting issues
    print("\nFixing linting issues...")
    run_command("ruff check --fix .")

    # Format code
    print("\nFormatting code...")
    run_command("ruff format .")

    print("\n✅ Formatting complete!")
    return 0


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/format_code.py [check|fix]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "check":
        sys.exit(check_formatting())
    elif command == "fix":
        sys.exit(fix_formatting())
    else:
        print(f"Unknown command: {command}")
        print("Usage: python scripts/format_code.py [check|fix]")
        sys.exit(1)


if __name__ == "__main__":
    main()