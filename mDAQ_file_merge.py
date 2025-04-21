# --- Necessary Imports for process_mdaq ---
import os
import csv
import logging
import time # For performance timing within the function
from tqdm import tqdm # For progress bar
from typing import Dict, List, Tuple

# --- Isolated mDAQ Processing Function ---

def process_mdaq(mdaq_folder: str, start_time_ms: int) -> Tuple[Dict[int, List[str]], int, int]:
    """
    Processes mDAQ data from a folder containing sequentially numbered CSV files.
    Calculates timestamps based on the initial BIOPAC start time and accumulated
    inter-sample intervals (ISI) from the mDAQ files.

    Args:
        mdaq_folder: Path to the folder containing mDAQ CSV files (e.g., 1.csv, 2.csv...).
        start_time_ms: The starting timestamp obtained from the BIOPAC file (milliseconds).

    Returns:
        A tuple containing:
        - mdaq_data (dict): Dictionary mapping calculated timestamps (int) to mDAQ data rows (list of str).
        - end_time_ms (int): The last calculated timestamp for mDAQ data (milliseconds).
        - processed_count (int): Number of data rows successfully processed and added to dict.

    Raises:
        NotADirectoryError: If mdaq_folder is not a valid directory.
        FileNotFoundError: If no suitable CSV files are found in the directory.
    """
    mdaq_data: Dict[int, List[str]] = {}
    processed_count = 0 # Initialize counter
    isi_accumulator_us = 0 # Accumulator for inter-sample intervals in microseconds
    # Note: Logging calls assume logging has been configured elsewhere if run standalone.
    logging.info(f'Starting mDAQ data processing in folder: {mdaq_folder}')
    start_process_time = time.time()

    if not os.path.isdir(mdaq_folder):
        raise NotADirectoryError(f"mDAQ folder not found or is not a directory: {mdaq_folder}")

    # Find CSV files named with digits (e.g., '1.csv', '10.csv')
    try:
        files = [f for f in os.listdir(mdaq_folder)
                 if f.endswith('.csv') and f[:-4].isdigit()]
        # Sort files numerically based on their names
        sorted_files = sorted(files, key=lambda x: int(x[:-4]))
    except Exception as e:
        logging.error(f"Error listing or sorting files in mdaq folder {mdaq_folder}: {e}")
        raise

    if not sorted_files:
        raise FileNotFoundError(f"No mDAQ CSV files (e.g., 1.csv, 2.csv) found in folder: {mdaq_folder}")

    logging.info(f"Found {len(sorted_files)} mDAQ files to process.")

    skipped_count = 0
    last_timestamp_ms = start_time_ms # Initialize with start time
    collision_check: Dict[int, int] = {} # To check for timestamp collisions

    # Process each file in numerical order
    for filename in tqdm(sorted_files, desc="Processing mDAQ files", unit="file"):
        file_path = os.path.join(mdaq_folder, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader) # Skip header row
                    logging.debug(f"mDAQ Header in {filename}: {header}")
                except StopIteration:
                    logging.warning(f"Skipping empty mDAQ file: {filename}")
                    continue # Skip empty files

                # Expected columns based on original code logic (though not strictly enforced later)
                # expected_columns = 16

                for row_idx, row in enumerate(reader):
                    # Check if enough columns exist for ISI access (index 11)
                    if len(row) < 12:
                        logging.warning(f"Skipping row {row_idx + 2} in {filename}: Too few columns ({len(row)}). Expected at least 12 for ISI. Line: {row}")
                        skipped_count += 1
                        continue

                    try:
                        # ISI is typically in column 12 (index 11)
                        isi_us = int(float(row[11]))
                        isi_accumulator_us += isi_us
                        # Calculate timestamp relative to BIOPAC start
                        timestamp_ms = start_time_ms + (isi_accumulator_us // 1000)

                        # Check for collisions (optional, for debugging)
                        if timestamp_ms in collision_check:
                            collision_check[timestamp_ms] += 1
                            if collision_check[timestamp_ms] == 2: # Log only first collision per timestamp
                                logging.warning(f"mDAQ timestamp collision detected for {timestamp_ms} ms in file {filename}, row approx {row_idx + 2}.")
                        else:
                            collision_check[timestamp_ms] = 1

                        last_timestamp_ms = timestamp_ms

                        # --- Data Extraction and Transformation ---
                        try:
                            ecg = row[0] # Index 0
                            # EDA calculation: Use raw value directly as per original code's assignment
                            eda_raw_str = row[1] # Index 1
                            # Optional: Could add a check here if eda_raw_str is numeric before assigning
                            eda = str(eda_raw_str) # Keep as string based on original logic

                        except IndexError: # Should not happen due to len check, but for safety
                            logging.warning(f"Skipping row {row_idx + 2} in {filename}: Missing ECG or EDA column. Row: {row}")
                            skipped_count += 1
                            continue

                        # Safely access other columns using checks for length
                        ir = row[2] if len(row) > 2 else ''
                        red = row[3] if len(row) > 3 else ''
                        acc_x = row[4] if len(row) > 4 else ''
                        acc_y = row[5] if len(row) > 5 else ''
                        acc_z = row[6] if len(row) > 6 else ''
                        gyr_x = row[7] if len(row) > 7 else ''
                        gyr_y = row[8] if len(row) > 8 else ''
                        gyr_z = row[9] if len(row) > 9 else ''
                        # Column 10 (index 9) seems unused based on output assembly
                        # Index 11 is ISI
                        batt = row[12] if len(row) > 12 else ''
                        rh = row[13] if len(row) > 13 else ''
                        amb_temp = row[14] if len(row) > 14 else ''
                        body_temp = row[15] if len(row) > 15 else ''

                        # Assemble the final data row in the desired order
                        values = [
                            ecg, eda, ir, red, acc_x, acc_y, acc_z,
                            gyr_x, gyr_y, gyr_z, batt, rh, amb_temp, body_temp
                        ]

                        # Store data with calculated timestamp
                        mdaq_data[timestamp_ms] = values

                    except (ValueError, IndexError) as e:
                        # Catch errors during ISI conversion or general access errors
                        logging.warning(f"Skipping row {row_idx + 2} in {filename}: Error processing ISI or accessing data. Error: {e}. Row: {row}")
                        skipped_count += 1
                        continue # Skip row on error

        except FileNotFoundError:
            # This specific file might disappear mid-processing (unlikely but possible)
            logging.error(f"mDAQ file disappeared during processing: {filename}")
        except Exception as e:
            # Catch general errors reading/processing a specific file
            logging.error(f"Error reading or processing mDAQ file {filename}: {e}")

    # Calculate final processed count based on the number of unique timestamps generated
    processed_count = len(mdaq_data)
    num_collisions = sum(count - 1 for count in collision_check.values() if count > 1)

    processing_time = time.time() - start_process_time
    logging.info(f'Finished mDAQ data processing. Time taken: {processing_time:.2f} seconds.')
    logging.info(f'mDAQ samples added to dictionary (unique timestamps): {processed_count}')
    logging.info(f'mDAQ rows skipped: {skipped_count}')
    if num_collisions > 0:
        logging.warning(f'mDAQ timestamp collisions detected (overwritten rows): {num_collisions}')
    else:
        logging.info('No mDAQ timestamp collisions detected.')

    if not mdaq_data:
        logging.warning("No mDAQ data was successfully processed.")
        # Return the original start time if no data was processed
        return mdaq_data, start_time_ms, 0

    logging.info(f"mDAQ data range: {start_time_ms} ms to {last_timestamp_ms} ms")
    return mdaq_data, last_timestamp_ms, processed_count

# --- Example Usage Placeholder (Optional) ---
# If you want to run this standalone, you would need to:
# 1. Configure the logging module (e.g., basicConfig)
# 2. Provide a valid 'mdaq_folder' path.
# 3. Provide a valid 'start_time_ms'.
#
# Example:
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#     my_mdaq_folder = "/path/to/your/mdaq_csvs" # CHANGE THIS
#     # Get start time from BIOPAC processing or set a known value
#     my_start_time = 1678886400000 # Example timestamp in ms
#
#     try:
#         data, end_ts, count = process_mdaq(my_mdaq_folder, my_start_time)
#         print(f"Processed {count} mDAQ data points.")
#         # print(data) # Optionally print the resulting dictionary
#     except Exception as e:
#         print(f"An error occurred: {e}")