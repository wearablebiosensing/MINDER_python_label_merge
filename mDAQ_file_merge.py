# --- Necessary Imports ---
import os
import csv
import logging
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

# --- GUI Function for Folder Selection ---

def select_mdaq_and_output_folders() -> Optional[Tuple[str, str]]:
    """
    Uses Tkinter dialogs to prompt the user to select the
    mDAQ folder and the output folder.

    Returns:
        A tuple containing the paths (mdaq_folder, output_folder)
        if both are selected, otherwise None.
    """
    root = tk.Tk()
    root.withdraw() # Hide the main Tkinter window

    messagebox.showinfo("Instructions", "Please select the required folders:\n1. Folder containing mDAQ .csv files\n2. Output folder for results")

    mdaq_folder = ""
    output_folder = ""

    # Select mDAQ folder
    mdaq_folder = filedialog.askdirectory(
        title="1. Select mDAQ folder containing CSV files"
    )
    if not mdaq_folder:
        messagebox.showerror("Error", "mDAQ folder selection cancelled.")
        root.destroy()
        return None # Indicate cancellation or error
    print(f"mDAQ folder selected: {mdaq_folder}")

    # Select Output folder
    output_folder = filedialog.askdirectory(
        title="2. Select Output folder for processed data"
    )
    if not output_folder:
        messagebox.showerror("Error", "Output folder selection cancelled.")
        root.destroy()
        return None # Indicate cancellation or error
    print(f"Output folder selected: {output_folder}")

    root.destroy() # Close the Tkinter root window
    return mdaq_folder, output_folder

# --- Isolated mDAQ Processing Function ---

def process_mdaq(mdaq_folder: str, start_time_ms: int) -> Tuple[Dict[int, List[str]], int, int]:
    """
    Processes mDAQ data from a folder containing sequentially numbered CSV files.
    Calculates timestamps based on the initial BIOPAC start time and accumulated
    inter-sample intervals (ISI) from the mDAQ files.

    Args:
        mdaq_folder: Path to the folder containing mDAQ CSV files (e.g., 1.csv, 2.csv...).
        start_time_ms: The starting timestamp obtained from the BIOPAC file (milliseconds).
                           Needs to be provided externally (e.g., hardcoded or from another source).

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
    # Note: Logging calls assume logging has been configured elsewhere.
    # Basic configuration is added in the __main__ block for demonstration.
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

                for row_idx, row in enumerate(reader):
                    if len(row) < 12:
                        logging.warning(f"Skipping row {row_idx + 2} in {filename}: Too few columns ({len(row)}). Expected at least 12 for ISI. Line: {row}")
                        skipped_count += 1
                        continue

                    try:
                        isi_us = int(float(row[11])) # ISI is in column 12 (index 11)
                        isi_accumulator_us += isi_us
                        timestamp_ms = start_time_ms + (isi_accumulator_us // 1000)

                        if timestamp_ms in collision_check:
                            collision_check[timestamp_ms] += 1
                            if collision_check[timestamp_ms] == 2:
                                logging.warning(f"mDAQ timestamp collision detected for {timestamp_ms} ms in file {filename}, row approx {row_idx + 2}.")
                        else:
                            collision_check[timestamp_ms] = 1

                        last_timestamp_ms = timestamp_ms

                        # --- Data Extraction ---
                        try:
                            ecg = row[0]
                            eda_raw_str = row[1]
                            eda = str(eda_raw_str) # Keep raw value as string
                        except IndexError:
                            logging.warning(f"Skipping row {row_idx + 2} in {filename}: Missing ECG or EDA column. Row: {row}")
                            skipped_count += 1
                            continue

                        ir = row[2] if len(row) > 2 else ''
                        red = row[3] if len(row) > 3 else ''
                        acc_x = row[4] if len(row) > 4 else ''
                        acc_y = row[5] if len(row) > 5 else ''
                        acc_z = row[6] if len(row) > 6 else ''
                        gyr_x = row[7] if len(row) > 7 else ''
                        gyr_y = row[8] if len(row) > 8 else ''
                        gyr_z = row[9] if len(row) > 9 else ''
                        batt = row[12] if len(row) > 12 else ''
                        rh = row[13] if len(row) > 13 else ''
                        amb_temp = row[14] if len(row) > 14 else ''
                        body_temp = row[15] if len(row) > 15 else ''

                        values = [
                            ecg, eda, ir, red, acc_x, acc_y, acc_z,
                            gyr_x, gyr_y, gyr_z, batt, rh, amb_temp, body_temp
                        ]
                        mdaq_data[timestamp_ms] = values

                    except (ValueError, IndexError) as e:
                        logging.warning(f"Skipping row {row_idx + 2} in {filename}: Error processing ISI or accessing data. Error: {e}. Row: {row}")
                        skipped_count += 1
                        continue

        except FileNotFoundError:
            logging.error(f"mDAQ file disappeared during processing: {filename}")
        except Exception as e:
            logging.error(f"Error reading or processing mDAQ file {filename}: {e}")

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
        return mdaq_data, start_time_ms, 0

    logging.info(f"mDAQ data range: {start_time_ms} ms to {last_timestamp_ms} ms")
    return mdaq_data, last_timestamp_ms, processed_count


# --- Main Execution Logic Example ---
if __name__ == "__main__":
    # --- Basic Logging Setup (for demonstration) ---
    # In a real application, you might want a more robust setup like in your original script
    log_format = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=[
        logging.StreamHandler() # Log to console
        # You could add a FileHandler here if needed, pointing to the selected output directory
    ])
    logging.info("--- Starting Script ---")

    selected_folders = None
    mdaq_folder_path = None
    output_folder_path = None

    try:
        # --- Use GUI to select folders ---
        selected_folders = select_mdaq_and_output_folders()

        if selected_folders:
            mdaq_folder_path, output_folder_path = selected_folders
            logging.info(f"Using mDAQ Folder: {mdaq_folder_path}")
            logging.info(f"Using Output Folder: {output_folder_path}")

            # --- IMPORTANT: Define the start time ---
            # This needs to come from somewhere, as it's not selected via the GUI here.
            # For example, hardcode it, read it from a file, or get it from prior BIOPAC processing.
            # Using an example value for demonstration:
            biopac_start_time_ms = 1678886400000 # Example: March 15, 2023 12:00:00 GMT
            logging.info(f"Using predefined start time: {biopac_start_time_ms} ms")

            # --- Process mDAQ Data ---
            mdaq_data, last_mdaq_ts, mdaq_count = process_mdaq(mdaq_folder_path, biopac_start_time_ms)

            logging.info(f"Successfully processed {mdaq_count} mDAQ data points.")
            logging.info(f"Last mDAQ timestamp: {last_mdaq_ts}")

            # --- Placeholder for Writing Data ---
            # You would add your data writing logic here, using the 'output_folder_path'
            # and the processed 'mdaq_data'.
            # Example:
            # output_file = os.path.join(output_folder_path, "processed_mdaq_output.csv")
            # logging.info(f"Writing output to {output_file}...")
            # with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            #     writer = csv.writer(outfile)
            #     # Add appropriate header based on 'values' list structure in process_mdaq
            #     header = ['timestamp_ms', 'ecg', 'eda', 'ir', 'red', 'acc_x', 'acc_y', 'acc_z',
            #               'gyr_x', 'gyr_y', 'gyr_z', 'batt%', 'relative_humidity', 'ambient_temp', 'body_temp']
            #     writer.writerow(header)
            #     for ts in sorted(mdaq_data.keys()):
            #         writer.writerow([ts] + mdaq_data[ts])
            # logging.info("Finished writing output.")

            messagebox.showinfo("Success", f"mDAQ processing complete.\nProcessed {mdaq_count} data points.\n\nOutput folder: {output_folder_path}")

        else:
            logging.warning("Folder selection cancelled or failed. Exiting.")
            messagebox.showwarning("Cancelled", "Folder selection was cancelled. Script will exit.")

    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        logging.error(f"File/Folder Error: {e}", exc_info=True)
        messagebox.showerror("Error", f"A file or folder error occurred:\n{e}\nPlease check the selected mDAQ folder.")
    except Exception as e:
        logging.exception("An unexpected error occurred during processing.")
        messagebox.showerror("Unexpected Error", f"An unexpected error occurred:\n{type(e).__name__}: {e}\n\nCheck logs for details.")
    finally:
        # Ensure Tkinter root is destroyed if it somehow still exists
        try:
            root = tk._default_root
            if root and root.winfo_exists():
                root.destroy()
        except Exception:
            pass # Ignore errors during cleanup
        logging.info("--- Script Finished ---")