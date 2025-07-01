#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Both Tasks
Easy-to-use script for judges to evaluate both Task A and Task B models.
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80)

def print_task_info():
    """Print information about both tasks"""
    print_header("FACE RECOGNITION COMPETITION - EVALUATION SUITE")

    print("\nTASK A - GENDER CLASSIFICATION")
    print("  Objective: Predict gender (Male/Female) from degraded face images")
    print("  Metrics:   Accuracy | Precision | Recall | F1-Score")

    print("\nTASK B - FACE RECOGNITION")
    print("  Objective: Assign face images to correct person identities")
    print("  Metrics:   Top-1 Accuracy | Macro-averaged F1-Score")

    print("\nThis script will run both evaluations and generate comprehensive reports.")
    print("="*80)

def run_task_a_evaluation(model_path, data_dir, output_dir, use_val_set=True):
    """Run Task A (Gender Classification) evaluation"""
    print_header("RUNNING TASK A - GENDER CLASSIFICATION EVALUATION")

    # Prepare command
    cmd = [
        sys.executable, "test_gender_model.py",
        "--model_path", model_path,
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--batch_size", "32",
        "--num_examples", "20"
    ]

    if use_val_set:
        cmd.append("--use_val_set")

    print(f"Command: {' '.join(cmd)}")
    print("\nRunning Task A evaluation...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Task A evaluation completed successfully!")
        if result.stdout:
            print("\nOutput:")
            print(result.stdout[-1000:])  # Show last 1000 characters
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Task A evaluation failed!")
        print(f"Error: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def run_task_b_evaluation(model_path, data_dir, output_dir, num_queries=1000):
    """Run Task B (Face Recognition) evaluation"""
    print_header("RUNNING TASK B - FACE RECOGNITION EVALUATION")

    # Prepare command
    cmd = [
        sys.executable, "test_model.py",
        "--model_path", model_path,
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--num_queries", str(num_queries),
        "--batch_size", "32"
    ]

    print(f"Command: {' '.join(cmd)}")
    print("\nRunning Task B evaluation...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Task B evaluation completed successfully!")
        if result.stdout:
            print("\nOutput:")
            print(result.stdout[-1000:])  # Show last 1000 characters
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Task B evaluation failed!")
        print(f"Error: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def generate_summary_report(task_a_output, task_b_output, summary_output):
    """Generate a combined summary report"""
    print_header("GENERATING SUMMARY REPORT")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open(summary_output, 'w') as f:
            f.write("FACE RECOGNITION COMPETITION - EVALUATION SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Evaluation Date: {timestamp}\n\n")

            # Task A Summary
            f.write("TASK A - GENDER CLASSIFICATION RESULTS\n")
            f.write("-" * 40 + "\n")

            task_a_metrics_file = os.path.join(task_a_output, "performance_metrics.txt")
            if os.path.exists(task_a_metrics_file):
                with open(task_a_metrics_file, 'r') as ta_file:
                    f.write(ta_file.read())
                f.write("\n")
            else:
                f.write("Task A results not found.\n\n")

            # Task B Summary
            f.write("\nTASK B - FACE RECOGNITION RESULTS\n")
            f.write("-" * 40 + "\n")

            task_b_metrics_file = os.path.join(task_b_output, "threshold_analysis.txt")
            if os.path.exists(task_b_metrics_file):
                with open(task_b_metrics_file, 'r') as tb_file:
                    # Read first few lines with the summary
                    lines = tb_file.readlines()
                    for line in lines[:10]:  # First 10 lines contain the summary
                        f.write(line)
                f.write("\n")
            else:
                f.write("Task B results not found.\n\n")

            f.write("\nEVALUATION CRITERIA:\n")
            f.write("-" * 20 + "\n")
            f.write("Task A: Accuracy | Precision | Recall | F1-Score\n")
            f.write("Task B: Top-1 Accuracy | Macro-averaged F1-Score\n")

        print(f"✅ Summary report generated: {summary_output}")
        return True

    except Exception as e:
        print(f"❌ Failed to generate summary report: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Evaluation for Face Recognition Competition')

    # Task A arguments
    parser.add_argument('--task_a_model', type=str, default='checkpoints/gender_model.pt',
                       help='Path to Task A (gender classification) model')
    parser.add_argument('--task_a_data', type=str, default='data/facecom/Task_A/',
                       help='Path to Task A data directory')

    # Task B arguments
    parser.add_argument('--task_b_model', type=str, default='checkpoints/final_model.pth',
                       help='Path to Task B (face recognition) model')
    parser.add_argument('--task_b_data', type=str, default='data/facecom/Task_B/',
                       help='Path to Task B data directory')

    # General arguments
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Base output directory for all results')
    parser.add_argument('--task_a_only', action='store_true',
                       help='Run only Task A evaluation')
    parser.add_argument('--task_b_only', action='store_true',
                       help='Run only Task B evaluation')
    parser.add_argument('--use_val_set', action='store_true', default=True,
                       help='Use validation set for Task A (default: True)')
    parser.add_argument('--num_queries', type=int, default=1000,
                       help='Number of queries for Task B evaluation')

    args = parser.parse_args()

    # Print task information
    print_task_info()

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = os.path.join(args.output_dir, f"evaluation_{timestamp}")
    task_a_output = os.path.join(base_output, "task_a_results")
    task_b_output = os.path.join(base_output, "task_b_results")

    os.makedirs(task_a_output, exist_ok=True)
    os.makedirs(task_b_output, exist_ok=True)

    print(f"\n📁 Results will be saved to: {base_output}")

    # Run evaluations
    task_a_success = True
    task_b_success = True

    if not args.task_b_only:
        # Check if Task A model exists
        if not os.path.exists(args.task_a_model):
            print(f"⚠️  Task A model not found: {args.task_a_model}")
            task_a_success = False
        else:
            task_a_success = run_task_a_evaluation(
                args.task_a_model,
                args.task_a_data,
                task_a_output,
                args.use_val_set
            )

    if not args.task_a_only:
        # Check if Task B model exists
        if not os.path.exists(args.task_b_model):
            print(f"⚠️  Task B model not found: {args.task_b_model}")
            task_b_success = False
        else:
            task_b_success = run_task_b_evaluation(
                args.task_b_model,
                args.task_b_data,
                task_b_output,
                args.num_queries
            )

    # Generate summary report
    summary_file = os.path.join(base_output, "EVALUATION_SUMMARY.txt")
    summary_success = generate_summary_report(task_a_output, task_b_output, summary_file)

    # Final summary
    print_header("EVALUATION COMPLETE")
    print(f"📊 Results saved to: {base_output}")

    if not args.task_b_only:
        print(f"Task A (Gender Classification): {'✅ SUCCESS' if task_a_success else '❌ FAILED'}")
    if not args.task_a_only:
        print(f"Task B (Face Recognition):      {'✅ SUCCESS' if task_b_success else '❌ FAILED'}")

    print(f"Summary Report: {'✅ GENERATED' if summary_success else '❌ FAILED'}")

    if summary_success:
        print(f"\n📋 Quick Summary: {summary_file}")
        print(f"📁 Detailed Results: {base_output}")

    print("\n🎯 EVALUATION CRITERIA:")
    print("   Task A: Accuracy | Precision | Recall | F1-Score")
    print("   Task B: Top-1 Accuracy | Macro-averaged F1-Score")
    print("="*80)

if __name__ == "__main__":
    main()
