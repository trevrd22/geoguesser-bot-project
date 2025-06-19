import os
import zipfile
from pathlib import Path
from tqdm import tqdm


def unzip_all_files(source_folder, destination_folder):
    """
    Unzips all ZIP files in the source folder and extracts them to the destination folder.
    Skips ZIPs that have already been extracted fully.
    """
    source_path = Path(source_folder)
    dest_path = Path(destination_folder)

    dest_path.mkdir(parents=True, exist_ok=True)

    total_files = 0
    successful_extractions = 0

    print(f"Searching for ZIP files in: {source_path}")
    print(f"Extracting to: {dest_path}")

    # Sort ZIP files by name
    zip_files = sorted(source_path.glob("*.zip"), key=lambda x: x.name)

    # Set the start index to skip already extracted files (e.g., skip 00–49)
    start_index = 25
    zip_files = zip_files[start_index:]
    total_files = len(zip_files)

    with tqdm(zip_files, desc="Overall progress", unit="file") as overall_pbar:
        for zip_file in overall_pbar:
            overall_pbar.set_postfix(file=zip_file.name[:20])
            try:
                extract_folder = dest_path / zip_file.stem

                # Skip extraction if already extracted
                if extract_folder.exists():
                    try:
                        with zipfile.ZipFile(zip_file, "r") as zip_ref:
                            zip_contents = set(zip_ref.namelist())

                            # Progress bar for counting extracted files
                            all_files = list(
                                p for p in extract_folder.rglob("*") if p.is_file()
                            )
                            extracted_contents = set()
                            with tqdm(
                                all_files,
                                desc=f"Comparing {zip_file.name[:15]}...",
                                leave=False,
                                unit="file",
                            ) as compare_pbar:
                                for p in compare_pbar:
                                    rel_path = str(
                                        p.relative_to(extract_folder)
                                    ).replace("\\", "/")
                                    extracted_contents.add(rel_path)

                            if zip_contents.issubset(extracted_contents):
                                print(f"\nSkipping {zip_file.name} (already extracted)")
                                successful_extractions += 1
                                continue
                    except Exception as e:
                        print(
                            f"\nWarning: Could not verify contents of {zip_file.name}: {str(e)}"
                        )

                # Extract
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    file_list = zip_ref.namelist()
                    with tqdm(
                        file_list,
                        desc=f"Extracting {zip_file.name[:15]}...",
                        unit="file",
                        leave=False,
                    ) as file_pbar:
                        for file in file_pbar:
                            try:
                                zip_ref.extract(file, extract_folder)
                            except Exception as e:
                                print(
                                    f"\nError extracting {file} from {zip_file.name}: {str(e)}"
                                )

                successful_extractions += 1
                print(
                    f"\nSuccessfully extracted {len(file_list)} files to: {extract_folder}"
                )

            except zipfile.BadZipFile:
                print(
                    f"\nError: {zip_file.name} is not a valid ZIP file or is corrupted"
                )
            except Exception as e:
                print(f"\nError processing {zip_file.name}: {str(e)}")

    print("\nExtraction complete!")
    print(f"Processed {total_files} ZIP files")
    print(f"Successfully extracted {successful_extractions} files")
    if total_files > successful_extractions:
        print(f"Failed to extract {total_files - successful_extractions} files")


if __name__ == "__main__":
    source_dir = input("Enter the source folder path containing ZIP files: ").strip()
    dest_dir = input("Enter the destination folder path for extraction: ").strip()
    unzip_all_files(source_dir, dest_dir)

# data\\external\\osv5m\\images\\train
# data\\raw\\unzippedtrain
