import pandas as pd
import os
import argparse
from vthree_utils import BinCreateParams

def csv_to_rst_table(df, title=None, headers=True, widths=None):
    """
    Convert a pandas DataFrame to an RST table string.
    
    Args:
        df: pandas DataFrame
        title: Optional table title
        headers: Whether to include headers
        widths: List of column widths (integers)
        
    Returns:
        str: RST table representation
    """
    # Format values for better readability
    df_formatted = df.copy()
    for col in df.select_dtypes(include=['float64']).columns:
        df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
    
    # Get column names and row values
    columns = df_formatted.columns
    rows = [list(row) for _, row in df_formatted.iterrows()]
    
    # Calculate column widths if not provided
    if widths is None:
        # Get max width of each column data
        col_widths = [max(len(str(x)) for x in df_formatted[col]) for col in columns]
        # Get width of column headers
        header_widths = [len(str(col)) for col in columns]
        # Take the maximum of data width and header width for each column
        widths = [max(col_widths[i], header_widths[i]) for i in range(len(columns))]
        
    # Create the table structure
    separator = " ".join("=" * width for width in widths)
    header_row = " ".join(str(col).ljust(widths[i]) for i, col in enumerate(columns))
    
    # Build the table
    rst_table = []
    
    if title:
        rst_table.append(f".. table:: {title}")
        rst_table.append("   :widths: auto")
        rst_table.append("")
        # Add 3 spaces before each line in the table
        indent = "   "
    else:
        indent = ""
    
    rst_table.append(f"{indent}{separator}")
    
    if headers:
        rst_table.append(f"{indent}{header_row}")
        rst_table.append(f"{indent}{separator}")
    
    for row in rows:
        row_str = " ".join(str(val).ljust(widths[i]) for i, val in enumerate(row))
        rst_table.append(f"{indent}{row_str}")
    
    rst_table.append(f"{indent}{separator}")
    
    return "\n".join(rst_table)

def run_data_table_rst(params):
    """
    Generate RST table from CSV data for the given parameters.
    """
    # Load the CSV file
    csv_path = f"{params.output_path}{params.region_id}_{params.sc_season_str}_{params.lead_int}.csv"
    df = pd.read_csv(csv_path)
    
    # Filter the data as needed
    filtered_df = df[df["td"] == 1]
    
    # Select the columns of interest
    filtered_df1 = filtered_df[
        [
            "x2d_leadtime",
            "trigger_value",
            "x2d_level",
            "obs_count",
            "hits",
            "misses",
            "FA",
            "CN",
            "hit_percentage",
        ]
    ]
    
    # Rename columns for better readability
    new_names = {
        "x2d_leadtime": "lead_time",
        "trigger_value": "Trigger",
        "obs_count": "odc",
        "x2d_level": "cat",
        "hit_percentage": "%hit",
    }
    dec_df = filtered_df1.rename(columns=new_names)
    
    # Convert percentage columns to proper format
    dec_df["Trigger"] = dec_df["Trigger"] * 100
    dec_df["Trigger"] = dec_df["Trigger"].apply(lambda x: round(x, 2))
    dec_df["%hit"] = dec_df["%hit"].apply(lambda x: round(x, 2))
    
    # Generate the RST table
    region_name = params.region_name_dict[params.region_id]
    table_title = f"Available triggers with >0.5 AUROC, >50% HR, <35% FAR for season {params.season_str}-{params.spi_prod_name} at region {region_name} with lead time {params.lead_int}"
    
    rst_table = csv_to_rst_table(dec_df, title=table_title)
    
    # Save the RST table to a file
    output_path = f"{params.output_path}{params.region_id}_{params.sc_season_str}_lt{params.lead_int}.rst"
    with open(output_path, "w") as f:
        f.write(rst_table)
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Generate RST tables from CSV data")
    parser.add_argument("--region-id", type=int, default=0, help="Region ID")
    parser.add_argument("--season-str", type=str, default="MAM", help="Season string")
    parser.add_argument("--lead-int", type=int, default=2, help="Lead time")
    parser.add_argument("--output-path", type=str, default="./output/", help="Output directory")
    args = parser.parse_args()
    
    # Create parameters object
    params = BinCreateParams(
        region_id=args.region_id,
        season_str=args.season_str,
        lead_int=args.lead_int,
        level="mod",
        region_name_dict={0: "Karamoja", 1: "Marsabit", 2: "Wajir"},
        spi_prod_name="spi3",
        data_path="./data/",
        output_path=args.output_path,
        spi4_data_path="",
        obs_netcdf_file="",
        fct_netcdf_file="",
        service_account_json="",
        gcs_file_url="",
        region_filter="kmj",
    )
    
    # Generate RST table
    output_file = run_data_table_rst(params)
    print(f"RST table saved to: {output_file}")

if __name__ == "__main__":
    main()
