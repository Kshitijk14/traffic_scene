import csv
from pathlib import Path

def save_object_log_csv(csv_path, fieldnames, data_row):
    """Appends a row to a CSV file, creating headers if the file doesn't exist."""
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    file_exists = Path(csv_path).is_file()

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data_row)
