#!/usr/bin/env python3
"""
Ruff formatting fix script for SAHI PR #1222
This script fixes all code formatting issues identified by ruff
"""

import os
import subprocess
import sys


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return None


def main():
    """Main function to fix ruff formatting issues."""
    print("🚀 Starting Ruff formatting fixes for SAHI PR #1222...")

    # Check if we're in the right directory
    if not os.path.exists("pyproject.toml"):
        print("❌ Error: pyproject.toml not found. Please run this script from the SAHI root directory.")
        sys.exit(1)

    # Step 1: Install ruff if not available
    print("\n📦 Step 1: Installing/updating ruff...")
    run_command("pip install --upgrade ruff", "Installing ruff")

    # Step 2: Check current ruff issues
    print("\n🔍 Step 2: Checking current ruff issues...")
    issues = run_command("ruff check .", "Checking ruff issues")
    if issues:
        print("Current ruff issues found:")
        print(issues)

    # Step 3: Fix formatting issues
    print("\n🔧 Step 3: Fixing formatting issues...")
    run_command("ruff format .", "Formatting code with ruff")

    # Step 4: Fix import sorting
    print("\n📚 Step 4: Fixing import sorting...")
    run_command("ruff check --fix .", "Fixing import and code issues")

    # Step 5: Verify fixes
    print("\n✅ Step 5: Verifying fixes...")
    final_check = run_command("ruff check .", "Final ruff check")

    if final_check and "All checks passed" in final_check:
        print("🎉 All ruff issues fixed successfully!")
    else:
        print("⚠️ Some ruff issues may remain. Please check manually.")
        if final_check:
            print("Remaining issues:")
            print(final_check)

    print("\n🚀 Ruff formatting fixes completed!")
    print("💡 Next steps:")
    print("   1. Review the changes")
    print("   2. Commit the fixes")
    print("   3. Push to trigger new CI runs")


if __name__ == "__main__":
    main()
