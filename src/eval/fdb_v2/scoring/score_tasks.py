import json
import argparse
from pathlib import Path
from collections import defaultdict

def summarize_task_metrics(root_dir):
    # Mapping your folders to the Table Columns
    folder_to_col = {
        "Correction": "Correction",
        "EntityTracking": "Entity",
        "Safety": "Safety"
    }
    
    stats = defaultdict(lambda: [0.0, 0])
    base_path = Path(root_dir)
    
    if not base_path.exists():
        print(f"Error: Path {root_dir} does not exist.")
        return

    # Iterate through Correction, EntityTracking, and Safety folders
    for folder_name, col_name in folder_to_col.items():
        task_folder = base_path / folder_name
        
        if not task_folder.exists():
            continue

        for json_file in task_folder.glob("*_processed.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                score = data.get("Task-specific score")
                if score is not None:
                    stats[col_name][0] += float(score)
                    stats[col_name][1] += 1
            except:
                continue

    # Print the table
    print(f"\n{'System':<15} | {'Correction':<10} | {'Entity':<10} | {'Safety':<10}")
    print("-" * 55)
    
    row = ["Result"] # Generic label
    for col in ["Correction", "Entity", "Safety"]:
        total, count = stats[col]
        avg = total / count if count > 0 else 0.0
        row.append(f"{avg:.2f}")
            
    print(f"{row[0]:<15} | {row[1]:<10} | {row[2]:<10} | {row[3]:<10}")
    print("-" * 55)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True)
    args = parser.parse_args()

    summarize_task_metrics(args.root_dir)