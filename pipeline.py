import subprocess
import os
import sys
from pathlib import Path

def run_script(script_name, working_dir=None):
    """Run a Python script and check for errors"""
    print(f"\n=== Running {script_name} ===")
    
    cmd = [sys.executable, script_name]
    if working_dir:
        original_dir = os.getcwd()
        os.chdir(working_dir)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if working_dir:
        os.chdir(original_dir)
        
    if result.returncode != 0:
        print(f"Error running {script_name}:")
        print(result.stderr)
        sys.exit(1)
    
    print(result.stdout)
    return True

def main():
    # Get base directory
    base_dir = Path(__file__).parent
    
    # Define task directories
    task1_dir = base_dir / "resources/task_1"
    task2_dir = base_dir / "resources/task_2"
    
    # Pipeline steps with their working directories
    steps = [
        # Task 1
        (task1_dir / "merge_data_sets.py", task1_dir),
        
        # Task 2
        (task2_dir / "reference_property_min_max.py", task2_dir),
        (task2_dir / "enrich_rfq.py", task2_dir),
        (task2_dir / "top3_experiment.py", task2_dir)
    ]
    
    # Run each step
    for script_path, working_dir in steps:
        run_script(script_path, working_dir)
    
    print("\n=== Pipeline completed successfully ===")
    print("Output files generated:")
    print("\nTask 1:")
    print("- merged_supplier_data.csv")
    print("\nTask 2:")
    print("- reference_properties_split.tsv")
    print("- rfq_enriched.csv")
    print("- top3_balanced.csv")
    print("- top3_dimension_focus.csv")
    print("- top3_grade_focus.csv")
    print("- top3_categorical_focus.csv")
    print("- similarity_analysis.png")
    print("- cluster_analysis.png")
    print("- rfq_clusters.csv")

if __name__ == "__main__":
    main()