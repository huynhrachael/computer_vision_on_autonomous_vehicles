import glob
import pandas as pd

csv_files = glob.glob("*.csv")

# Read and concatenate all CSVs
combined_df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

# Save to a single CSV file
combined_df.to_csv("combined_output.csv", index=False)
print(f"Combined {len(csv_files)} files into combined_output.csv")