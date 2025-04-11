import pandas as pd

# Define the files and their specific rows
file_rows = {
    '0_jja_2.csv': [5, 74, 112],
    '0_jja_3.csv': [14, 83, 109],
    '0_jja_4.csv': [7, 96, 140]
}

# Process each file
for filename, rows in file_rows.items():
    # Read the CSV file
    df = pd.read_csv(f'output/{filename}')
    
    # Add 'td' column with default value 0
    df['td'] = ''
    
    # Set value to 1 for specified rows
    for row_num in rows:
        if row_num < len(df):
            df.loc[row_num, 'td'] = 1
    
    # Save the modified dataframe back to CSV
    df.to_csv(f'output/{filename}', index=False)
    
    print(f"Updated {filename} with td=1 for rows {rows}")

