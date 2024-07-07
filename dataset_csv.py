import os
import pandas as pd
import argparse
from tqdm import tqdm


def generate_csv(lower_bound, upper_bound, separator='\t'):
    image_list = []

    for i in tqdm(range(lower_bound, upper_bound + 1)):
        folder_name = f"/hdd5/zhiqi2/datasets/cc12m/{i:05d}"
        if os.path.exists(folder_name):
            for file in os.listdir(folder_name):
                if file.endswith(".jpg"):
                    image_path = folder_name + "/" + file
                    title_path = os.path.splitext(image_path)[0] + ".txt"

                    if os.path.exists(title_path):
                        with open(title_path, 'r') as title_file:
                            title = title_file.read().strip()
                    else:
                        print(f"Title file '{title_path}' not found.")

                    image_list.append({"filepath": image_path, "title": title})

    df = pd.DataFrame(image_list)
    number_of_images = len(image_list)
    csv_filename = f"{lower_bound:05d}-{upper_bound:05d}-{number_of_images}.csv"
    df.to_csv(csv_filename, sep=separator, index=False)

    print(f"CSV file '{csv_filename}' has been generated with {number_of_images} images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a CSV file listing image file paths and titles.")

    parser.add_argument("--lb", type=int, help="Lower bound for folder names, e.g., 00000.")
    parser.add_argument("--ub", type=int, help="Upper bound for folder names, e.g., 10000.")
    parser.add_argument("--separator", type=str, default='\t', help="Separator for the CSV file (default is tab).")

    args = parser.parse_args()

    generate_csv(args.lb, args.ub, args.separator)
