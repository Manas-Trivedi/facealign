#!/usr/bin/env python3
"""
Test Runner - Face Recognition & Gender Classification
Run comprehensive tests and generate results for both models.

Usage:
  python run_all_tests.py                    # Comprehensive analysis (default)
  python run_all_tests.py --official         # Official competition evaluation
  python run_all_tests.py --both             # Both comprehensive and official
"""

import os
import sys
import subprocess
import time
import argparse
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
    parser = argparse.ArgumentParser(description='Comprehensive Model Testing Suite')
    parser.add_argument('--official', action='store_true',
                       help='Run official competition evaluation (for judges)')
    parser.add_argument('--both', action='store_true',
                       help='Run both comprehensive analysis and official evaluation')
    args = parser.parse_args()

    # Determine what to run
    if args.official:
        mode = "OFFICIAL COMPETITION EVALUATION"
        run_comprehensive = False
        run_official = True
    elif args.both:
        mode = "COMPREHENSIVE + OFFICIAL EVALUATION"
        run_comprehensive = True
        run_official = True
    else:
        mode = "COMPREHENSIVE ANALYSIS"
        run_comprehensive = True
        run_official = False

    print(f"{mode} SUITE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Change to project directory
    project_dir = "/Users/manastrivedi/Code/facealign"
    os.chdir(project_dir)
    print(f"Working directory: {project_dir}")

    # Build test list based on mode
    tests = []

    if run_comprehensive:
        tests.extend([
            ("python test_gender_model.py", "Gender Classification - Comprehensive Analysis"),
            ("python test_model.py", "Face Recognition - Comprehensive Analysis"),
        ])

    if run_official:
        tests.append(("python run_evaluation.py", "Official Competition Evaluation"))

    print(f"\nRunning {len(tests)} test suite(s)...")
    if run_official:
        print("📊 Official evaluation metrics: Task A (Accuracy|Precision|Recall|F1) | Task B (Top-1 Accuracy|Macro F1)")

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
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {status} - {description}")

    print()

    if successful == total:
        print("🎉 All tests completed successfully!")
        print()
        print("📁 Results saved to:")
        if run_comprehensive:
            print("  - test_results_gender/ (Gender classification analysis)")
            print("  - test_results/ (Face recognition analysis)")
        if run_official:
            print("  - evaluation_results/ (Official competition evaluation)")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
