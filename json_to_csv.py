import os
import json
import pandas as pd
import argparse
import collections


def read_json_files_to_excel(folder_path):

    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)

            if "rai_jailbreak_result" not in file_name:
                continue

            # Read the JSON file
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    json_data = json.load(f)

                    # Flatten the JSON content into a single row with columns
                    arr = collections.defaultdict(list)

                    json_data = (
                        json_data["rows"]
                        if "jailbreak" not in file_name
                        else json_data[0]["rows"]
                    )
                    for row in json_data:
                        for key, val in row.items():
                            arr[key].append(val)

                    df = pd.DataFrame(arr)
                    # Save the DataFrame to an Excel file
                    output_file_name = file_name.split(".")[0] + ".xlsx"
                    df.to_excel(output_file_name, index=False)
                    print(f"Data saved to {output_file_name}")

                except Exception as e:
                    print(f"Error reading {file_name}: {e}")

                # Convert the list of dictionaries to a DataFrame


def main():

    print("Homie, we are processing the json")

    args = argparse.ArgumentParser()

    args.add_argument(
        "-f",
        "--folder",
        help="the folder of the path",
    )

    parsed_args = args.parse_args()

    # Call the function to process the JSON files and save to Excel
    read_json_files_to_excel(parsed_args.folder)


if __name__ == "__main__":
    main()
