#!/usr/bin/env python3
"""
Test Runner - Face Recognition & Gender Classification
Run comprehensive tests and generate results for both models.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_command(cmd, description):
    """Run a command and handle output"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")

    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, check=True,
                              capture_output=False, text=True)
        elapsed = time.time() - start_time
        print(f"COMPLETED in {elapsed:.1f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"FAILED after {elapsed:.1f} seconds")
        print(f"Error: {e}")
        return False

def main():
    print("COMPREHENSIVE MODEL TESTING SUITE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Change to project directory
    project_dir = "/Users/manastrivedi/Code/facealign"
    os.chdir(project_dir)
    print(f"Working directory: {project_dir}")

    # Test configuration
    tests = [
        ("python test_gender_model.py", "Gender Classification - Comprehensive Analysis"),
        ("python test_model.py", "Face Recognition - Comprehensive Analysis"),
    ]

    print(f"\nRunning {len(tests)} comprehensive test suites...")

    # Run all tests
    results = []
    total_start = time.time()

    for cmd, description in tests:
        success = run_command(cmd, description)
        results.append((description, success))

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print("TESTING SUMMARY")
    print(f"{'='*60}")

    successful = sum(1 for _, success in results if success)
    total = len(results)

    print(f"Total tests run: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Total time: {total_elapsed:.1f} seconds")
    print()

    for description, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"  {status} - {description}")

    print()

    if successful == total:
        print("All tests completed successfully!")
        print()
        print("Results saved to:")
        print("  - test_results_gender/ (Gender classification)")
        print("  - test_results/ (Face recognition)")
    else:
        print("Some tests failed. Check the output above for details.")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
