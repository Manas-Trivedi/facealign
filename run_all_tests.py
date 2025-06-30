#!/usr/bin/env python3
"""
Quick Test Runner - Face Recognition & Gender Classification
Run all tests and generate comprehensive results for both models.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_command(cmd, description):
    """Run a command and handle output"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")

    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, check=True,
                              capture_output=False, text=True)
        elapsed = time.time() - start_time
        print(f"✅ Completed in {elapsed:.1f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ Failed after {elapsed:.1f} seconds")
        print(f"Error: {e}")
        return False

def main():
    print("🎯 COMPREHENSIVE MODEL TESTING SUITE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Change to project directory
    project_dir = "/Users/manastrivedi/Code/facealign"
    os.chdir(project_dir)
    print(f"Working directory: {project_dir}")

    # List of tests to run
    tests = [
        ("python test_gender_model_simple.py", "Gender Classification - Quick Test"),
        ("python demo_gender.py", "Gender Classification - Demo Predictions"),
        ("python test_model_simple.py", "Face Recognition - Quick Test"),
        ("python show_predictions.py", "Face Recognition - Demo Predictions"),
    ]

    comprehensive_tests = [
        ("python test_gender_model.py", "Gender Classification - Comprehensive Analysis"),
        ("python test_model.py", "Face Recognition - Comprehensive Analysis"),
    ]

    # Ask user if they want comprehensive or quick tests
    print("\nSelect testing mode:")
    print("1. Quick tests only (recommended)")
    print("2. Comprehensive tests (takes longer)")

    choice = input("\nEnter choice (1 or 2, default=1): ").strip()

    if choice == "2":
        all_tests = tests + comprehensive_tests
        print("\n🔍 Running COMPREHENSIVE test suite...")
    else:
        all_tests = tests
        print("\n⚡ Running QUICK test suite...")

    # Run all tests
    results = []
    total_start = time.time()

    for cmd, description in all_tests:
        success = run_command(cmd, description)
        results.append((description, success))

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print("📊 TESTING SUMMARY")
    print(f"{'='*60}")

    successful = sum(1 for _, success in results if success)
    total = len(results)

    print(f"Total tests run: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Total time: {total_elapsed:.1f} seconds")
    print()

    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {status} - {description}")

    print()

    if successful == total:
        print("🎉 All tests completed successfully!")
        print()
        print("📁 Results saved to:")
        print("  - test_results_gender/ (Gender classification)")
        print("  - test_results/ (Face recognition)")
        print("  - FINAL_RESULTS_SUMMARY.md (Complete summary)")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
