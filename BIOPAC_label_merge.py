import os
import csv
import logging
import psutil
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from tqdm import tqdm
import bisect
import time
from typing import Dict, List, Tuple, Optional
import pandas as pd # Added for robust CSV and Excel reading

def find_nearest_timestamp(target: int, timestamps: List[int], window_ms: int = 1000) -> Optional[int]:
    """Find first timestamp that is either exact match or next available within window."""
    if not timestamps:
        return None
        
    idx = bisect.bisect_left(timestamps, target)
    
    # Return exact match or next available timestamp within window
    if idx < len(timestamps) and timestamps[idx] - target <= window_ms:
        return timestamps[idx]
        
    return None

def write_data_efficient(start_time_ms: int, end_time_ms: int, 
                         biopac_data: Dict, labels: Dict, 
                         output_dir: str, start_all_time: float) -> None: # Added start_all_time
    """Optimized data writing with label matching for BIOPAC device."""
    os.makedirs(output_dir, exist_ok=True)
    logging.info('Starting optimized data writing process')

    # Pre-sort timestamps
    biopac_timestamps = sorted(biopac_data.keys())
    
    # Process labels and track assignments
    label_assignments = []
    biopac_label_matches = {}
    
    for label_ts in sorted(labels.keys()):
        label = labels[label_ts]
        biopac_match = find_nearest_timestamp(label_ts, biopac_timestamps)
        
        if biopac_match:
            assignment = {
                'original_ts': label_ts,
                'label': label,
                'biopac_ts': biopac_match,
                'biopac_row': biopac_timestamps.index(biopac_match)+2 # +2 for header and 1-based indexing
            }
            label_assignments.append(assignment)
            biopac_label_matches[biopac_match] = label

    # Headers
    biopac_header = ['timestamp_ms', 'ECG', 'PPG', 'SKT', 'EDA', 'label']

    # Write data efficiently
    with open(os.path.join(output_dir, 'biopac_labels.csv'), 'w', newline='') as bf, \
         open(os.path.join(output_dir, 'label_assignments.csv'), 'w', newline='') as lf:

        writers = {
            'biopac': csv.writer(bf),
            'labels': csv.writer(lf)
        }

        # Write headers
        writers['biopac'].writerow(biopac_header)
        writers['labels'].writerow(['label', 'original_timestamp_ms', 'biopac_timestamp_ms', 
                                    'biopac_row', 'time_diff_ms'])

        # Write BIOPAC data with labels
        for ts in biopac_timestamps:
            values = biopac_data[ts]
            label = biopac_label_matches.get(ts, '')
            writers['biopac'].writerow([ts] + values + [label])

        # Write label assignments
        for assign in label_assignments:
            time_diff = abs(assign['original_ts'] - assign['biopac_ts'])
            writers['labels'].writerow([
                assign['label'],
                assign['original_ts'],
                assign['biopac_ts'],
                assign['biopac_row'],
                time_diff
            ])

    # Write performance metrics
    diagnostic_data = {
        'Processing Statistics': {
            'Total Processing Time': f"{time.time() - start_all_time:.2f}s", # Use passed start_all_time
            'Peak Memory Usage': f"{psutil.Process().memory_info().rss / (1024*1024):.2f}MB",
            'BIOPAC Points': len(biopac_data),
            'Labels Processed': len(labels),
            'Labels Matched': len(label_assignments)
        }
    }

    with open(os.path.join(output_dir, 'performance_metrics.txt'), 'w') as f:
        for section, metrics in diagnostic_data.items():
            f.write(f"\n{section}:\n")
            f.write("="*50 + "\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
    logging.info(f"Performance metrics written to {os.path.join(output_dir, 'performance_metrics.txt')}")

def process_labels(label_file: str, time_offset: int = 0) -> Tuple[Dict[int, str], Dict[str, str]]:
    """
    Process label file, attempting to read as CSV or Excel.
    Extracts labels and metadata.
    """
    labels: Dict[int, str] = {}
    metadata: Dict[str, str] = {}
    logging.info(f'Starting label data processing for: {label_file}')
    processing_start_time = time.time()
    
    raw_data_rows: List[List[str]] = []

    try:
        # Attempt 1: Try to read as CSV using pandas (more robust for various delimiters and issues)
        try:
            logging.info(f"Attempting to read '{label_file}' as CSV with pandas.")
            # Use utf-8-sig to handle potential Byte Order Mark (BOM)
            # header=None to treat the first row as data, to be consistent with how original code processes idx==0
            df = pd.read_csv(label_file, header=None, on_bad_lines='warn', encoding='utf-8-sig', dtype=str, keep_default_na=False)
            
            # Check if parsing resulted in very few columns (e.g. everything in one column)
            # This can happen if an Excel file (which is a zip archive) is read as CSV.
            if df.shape[1] < 2 and len(df) > 1: 
                logging.warning(
                    f"Pandas CSV read of '{label_file}' resulted in {df.shape[1]} column(s). "
                    "This might indicate it's not a standard CSV or has an unusual delimiter. "
                    "Will try reading as Excel."
                )
                raise pd.errors.ParserError("Potentially not a CSV or malformed, leading to few columns.")
            
            raw_data_rows = df.values.tolist()
            logging.info(f"Successfully read '{label_file}' as CSV using pandas. {len(raw_data_rows)} rows found.")

        except (pd.errors.ParserError, UnicodeDecodeError, FileNotFoundError, Exception) as e_csv:
            logging.warning(f"Pandas CSV read failed for '{label_file}': {e_csv}. Attempting to read as Excel.")
            try:
                # Attempt 2: Try to read as Excel using pandas
                # header=None ensures the first row is read as data
                # Make sure to install openpyxl: pip install openpyxl
                df = pd.read_excel(label_file, engine='openpyxl', header=None, dtype=str, keep_default_na=False)
                raw_data_rows = df.values.tolist()
                logging.info(f"Successfully read '{label_file}' as Excel using pandas (openpyxl engine). {len(raw_data_rows)} rows found.")
            except Exception as e_excel:
                logging.error(f"Failed to read '{label_file}' as Excel after CSV attempt failed: {e_excel}")
                # Fallback: Try python's built-in csv.reader as a last resort
                logging.info(f"Falling back to built-in csv.reader for '{label_file}'.")
                try:
                    with open(label_file, 'r', encoding='utf-8-sig', newline='') as f_fallback:
                        reader = csv.reader(f_fallback)
                        raw_data_rows = list(reader)
                    logging.info(f"Successfully read '{label_file}' with built-in csv.reader. {len(raw_data_rows)} rows found.")
                except Exception as e_fallback_csv:
                    logging.error(f"Built-in csv.reader also failed for '{label_file}': {e_fallback_csv}")
                    raise ValueError(
                        f"Could not process label file '{label_file}'. Tried pandas CSV, pandas Excel, and built-in CSV reader."
                    ) from e_excel # Chain the exception

        if not raw_data_rows:
            logging.warning(f"No data extracted from label file: {label_file}")
            return labels, metadata

        # Original processing logic, now using raw_data_rows (list of lists of strings)
        for idx, row in enumerate(raw_data_rows):
            if not row or not any(str(cell).strip() for cell in row): # Skip empty or effectively empty rows
                logging.debug(f"Skipping empty or blank row {idx+1}: {row}")
                continue
            
            try:
                # Ensure timestamp is treated as string first for float conversion, and strip whitespace
                timestamp_str = str(row[0]).strip() 
                
                # Handle header row for metadata and skip for labels dict
                if idx == 0: # Check the first row
                    is_header = False
                    try:
                        # Attempt to convert the first cell to float. 
                        # If it works, it's likely a data row, not a typical string header.
                        float(timestamp_str) 
                    except ValueError:
                        # If float conversion fails, it's likely a string header like "Timestamp".
                        is_header = True 
                    
                    if is_header:
                        logging.info(f"Processing row {idx+1} as potential header row for metadata: {row}")
                        if len(row) >= 6: # Original condition for metadata extraction
                            metadata = {
                                'device': str(row[2]).strip(),
                                'session': str(row[3]).strip(),
                                'subject': str(row[4]).strip(),
                                'trial': str(row[5]).strip() 
                            }
                            logging.info(f"Extracted metadata: {metadata}")
                        else:
                            logging.warning(
                                f"Header row {idx+1} found but not enough columns for full metadata "
                                f"(expected >=6, got {len(row)}): {row}"
                            )
                        continue # Move to the next row, don't process this header as a label entry

                # Process as a data row (this could be the first row if it wasn't identified as a string header)
                if not timestamp_str: # Ensure timestamp isn't empty after potential header skip
                    logging.warning(f"Skipping row {idx+1} due to empty timestamp string: {row}")
                    continue

                timestamp_ms = int(float(timestamp_str)) + time_offset
                # Ensure label_text is taken from row[1] if available, otherwise empty string
                label_text = str(row[1]).strip() if len(row) > 1 else ""
                labels[timestamp_ms] = label_text

            except ValueError as ve:
                logging.warning(f"Skipping row {idx+1} in '{label_file}' due to ValueError: {ve}. Row content: {row}")
                continue # Skip to the next row
            except IndexError as ie:
                # This means the row doesn't have enough columns (e.g., missing label column)
                logging.warning(f"Skipping row {idx+1} in '{label_file}' due to IndexError (likely missing columns): {ie}. Row content: {row}")
                # If we at least got a timestamp string that seems valid, record it with an empty label
                if timestamp_str: 
                    try:
                        timestamp_ms = int(float(timestamp_str)) + time_offset
                        labels[timestamp_ms] = "" # Assign empty label
                        logging.info(f"Row {idx+1} had a timestamp but was missing other fields. Recorded with empty label.")
                    except ValueError: 
                        logging.warning(f"Timestamp '{timestamp_str}' in row {idx+1} was also invalid after an IndexError.")
                continue # Skip to the next row
        
    except Exception as e:
        logging.error(f"A critical error occurred while processing label file {label_file}: {e}", exc_info=True)
        raise 

    processing_duration = time.time() - processing_start_time
    logging.info(f'Finished label data processing for {label_file}. Time taken: {processing_duration:.2f} seconds.')
    logging.info(f'Labels processed: {len(labels)}')
    if metadata:
        logging.info(f'Metadata found: {metadata}')
    else:
        logging.info('No metadata extracted from label file (or header not in expected format/length).')
    return labels, metadata

def process_biopac(biopac_file: str) -> Tuple[int, int, Dict[int, List[str]]]:
    """
    Process BIOPAC file with dynamic channel mapping based on header.
    File format expectations:
    - Header contains "Recording on: YYYY-MM-DD HH:MM:SS.fff" (case-insensitive)
    - Sample rate in "X msec/sample" format (case-insensitive)
    - Data header line starts with "sec,CH" (case-insensitive)
    - Required channels (case-insensitive): CH2 (for ECG), CH3 (for PPG), CH7 (for SKT), CH16 (for EDA)
    """
    biopac_data: Dict[int, List[str]] = {}
    logging.info(f'Starting BIOPAC data processing for: {biopac_file}')
    processing_start_time = time.time()

    try:
        with open(biopac_file, 'r', encoding='utf-8-sig') as f: # Added encoding
            lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"BIOPAC file not found: {biopac_file}")
        raise
    except Exception as e:
        logging.error(f"Error reading BIOPAC file {biopac_file}: {e}")
        raise

    # Parse sample rate
    sample_rate_line = next((line for line in lines if "msec/sample" in line.lower()), None)
    if not sample_rate_line:
        raise ValueError(f"Could not find sample rate (msec/sample) in BIOPAC file header: {biopac_file}")
    
    try:
        sample_rate_ms = float(sample_rate_line.split()[0])
        if sample_rate_ms <= 0:
            raise ValueError(f"Sample rate must be positive. Found: {sample_rate_ms}")
        logging.info(f"Detected sample rate: {sample_rate_ms} msec/sample")
    except (ValueError, IndexError) as e:
        raise ValueError(f"Failed to parse a valid sample rate from line: '{sample_rate_line.strip()}' in {biopac_file}") from e

    # Parse recording timestamp
    recording_line = next((line for line in lines if line.lower().startswith('recording on:')), None)
    if not recording_line:
        raise ValueError(f"Could not find 'Recording on:' timestamp in BIOPAC file: {biopac_file}")

    time_str = recording_line.split(':', 1)[1].strip() # Use maxsplit=1 for robustness
    try:
        start_datetime = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        try:
            start_datetime = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError as e_dt:
            raise ValueError(f"Failed to parse recording timestamp '{time_str}' in {biopac_file}. Expected YYYY-MM-DD HH:MM:SS[.fff]") from e_dt
    start_time_ms = int(start_datetime.timestamp() * 1000)

    # Find data section and parse channel headers
    data_header_idx = None
    channel_indices: Dict[str, int] = {} # To store CH_NAME -> column_index

    for idx, line in enumerate(lines):
        if line.lower().startswith('sec,ch'):
            data_header_idx = idx
            header_content = line.strip()
            raw_headers = [h.strip().upper() for h in header_content.split(',')]
            for i, header_name in enumerate(raw_headers):
                channel_indices[header_name] = i # Store all found CH headers and their indices
            logging.info(f"Found BIOPAC data header: {header_content}")
            logging.info(f"Parsed channel indices: {channel_indices}")
            break
    
    if data_header_idx is None:
        raise ValueError(f"Could not find data header line starting with 'sec,CH' in BIOPAC file: {biopac_file}")

    # Define which signals we want and which BIOPAC channels they map to
    # The output order in biopac_labels.csv is fixed: ECG, PPG, SKT, EDA
    signal_to_channel_map = { 
        'ECG': 'CH2', 
        'PPG': 'CH3', 
        'SKT': 'CH7', 
        'EDA': 'CH16'
    }
    
    ordered_column_indices_to_extract: List[int] = []
    missing_channels = []
    for signal_name, ch_designator in signal_to_channel_map.items():
        if ch_designator in channel_indices:
            ordered_column_indices_to_extract.append(channel_indices[ch_designator])
        else:
            missing_channels.append(f"{signal_name} (expected {ch_designator})")
    
    if missing_channels:
        raise ValueError(
            f"Missing required channels in BIOPAC header. Needed: {', '.join(missing_channels)}. "
            f"Found headers: {list(channel_indices.keys())} in file {biopac_file}"
        )
    logging.info(f"Column indices to extract for (ECG, PPG, SKT, EDA): {ordered_column_indices_to_extract}")

    # Determine where actual data lines start
    data_lines_start_from = data_header_idx + 1
    if data_lines_start_from < len(lines):
        next_line_content = lines[data_lines_start_from].strip()
        # Check for a "Samples" line, typical in some Biopac exports
        # Corrected split_whitespace() to split()
        if "samples" in next_line_content.lower() and next_line_content.split() and next_line_content.split()[0].isdigit():
            logging.info(f"Skipping sample count line: '{next_line_content}'")
            data_lines_start_from += 1
    
    current_timestamp_ms = start_time_ms
    num_data_lines_processed = 0

    for line_num_rel, line_content in enumerate(tqdm(lines[data_lines_start_from:], desc="Processing BIOPAC data", unit="line")):
        actual_line_num = data_lines_start_from + line_num_rel + 1 # For logging
        try:
            values_str = line_content.strip().split(',')
            if not values_str or not values_str[0]: # Skip empty lines or lines without a first value (time)
                logging.debug(f"Skipping empty/invalid data line {actual_line_num}: '{line_content.strip()}'")
                continue

            # Extract the required channel data using the determined column indices
            extracted_values = [values_str[col_idx] for col_idx in ordered_column_indices_to_extract]
            
            biopac_data[current_timestamp_ms] = extracted_values
            current_timestamp_ms += int(sample_rate_ms) 
            num_data_lines_processed +=1

        except IndexError:
            logging.warning(
                f"IndexError on BIOPAC data line {actual_line_num}: '{line_content.strip()}'. "
                f"Expected at least {max(ordered_column_indices_to_extract) + 1} columns based on header. "
                f"Skipping line."
            )
            continue
        except ValueError as e_val: # Catch issues like converting sample_rate_ms to int if it's bad
             logging.warning(
                f"ValueError on BIOPAC data line {actual_line_num}: '{line_content.strip()}'. "
                f"Error: {e_val}. Skipping line."
            )
             continue


    end_timestamp_ms = start_time_ms # Default if no data
    if num_data_lines_processed > 0:
        end_timestamp_ms = current_timestamp_ms - int(sample_rate_ms) # Timestamp of the last processed sample

    total_duration_calc_ms = (num_data_lines_processed * sample_rate_ms) if num_data_lines_processed > 0 else 0
    
    logging.info('BIOPAC Processing Summary:')
    logging.info(f'  File: {biopac_file}')
    logging.info(f'  Start time: {start_datetime} (Epoch ms: {start_time_ms})')
    logging.info(f'  End time of data: (Epoch ms: {end_timestamp_ms})')
    logging.info(f'  Sample rate: {sample_rate_ms} ms ({1000/sample_rate_ms:.2f} Hz)')
    logging.info(f'  Data lines processed: {num_data_lines_processed}')
    logging.info(f'  Calculated duration from processed data: {total_duration_calc_ms/1000:.2f} seconds')
    logging.info(f'  Processing time for this file: {time.time() - processing_start_time:.2f} seconds')
    
    if num_data_lines_processed == 0 and len(lines) > data_lines_start_from:
        logging.warning(f"Processed 0 data lines from BIOPAC file, but file had content after header. Check data format and required channels (CH2, CH3, CH7, CH16).")

    return start_time_ms, end_timestamp_ms, biopac_data


def setup_logger(output_dir: str) -> None:
    """Initializes logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, 'processing_performance.log')
    
    # Remove existing handlers to avoid duplicate logs if called multiple times
    # This is important if setup_logger is called more than once (e.g. after renaming output_dir)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=log_file_path,
        filemode='w', # Overwrite log file each run for this session's output
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        level=logging.INFO # Log INFO level and above to file
    )
    
    # Add a handler for console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO) # Set level for console output
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler) # Add console handler to root logger
    
    logging.info(f"Logger initialized. Log file: {log_file_path}")


def select_files() -> Tuple[str, str, str]:
    """Select input and output files/folders using Tkinter dialogs with guidance."""
    root = tk.Tk()
    root.withdraw() # Hide the main Tkinter window

    messagebox.showinfo("File Selection Step 1/3", "Please select the BIOPAC data file (usually a .txt file).")
    biopac_file = filedialog.askopenfilename(
        title="Select BIOPAC data file (.txt)",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    if not biopac_file:
        messagebox.showerror("Error", "BIOPAC file not selected. Exiting.")
        raise FileNotFoundError("BIOPAC file not selected.")

    messagebox.showinfo("File Selection Step 2/3", "Please select the corresponding label file (e.g., .csv, .xlsx, .txt).")
    label_file = filedialog.askopenfilename(
        title="Select label file (.csv, .xlsx, .txt)",
        filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("Text files", "*.txt"), ("All files", "*.*")]
    )
    if not label_file:
        messagebox.showerror("Error", "Label file not selected. Exiting.")
        raise FileNotFoundError("Label file not selected.")
        
    messagebox.showinfo("File Selection Step 3/3", "Please select an output folder where the processed files will be saved.")
    output_folder = filedialog.askdirectory(
        title="Select output folder for processed data"
    )
    if not output_folder:
        messagebox.showerror("Error", "Output folder not selected. Exiting.")
        raise NotADirectoryError("Output folder not selected.")

    return biopac_file, label_file, output_folder

def main() -> None:
    """Main function to orchestrate file processing and data writing."""
    start_all: float = time.time() 
    # output_dir_for_log is used to inform user where to find logs if an error occurs VERY early
    # before the final output_dir is established.
    output_dir_for_log: Optional[str] = None 
    current_output_dir: Optional[str] = None


    try:
        # --- 1. Select Files ---
        biopac_file, label_file, initial_output_folder = select_files()
        
        # --- 2. Initial Output Directory & Logger Setup ---
        # Create a preliminary, timestamped output directory. This will be the log location
        # if metadata processing fails or isn't available to make a more specific name.
        timestamp_for_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_output_dir = os.path.join(initial_output_folder, f"biopac_merge_run_{timestamp_for_folder}")
        output_dir_for_log = current_output_dir # For error messages if setup fails
        os.makedirs(current_output_dir, exist_ok=True)
        
        setup_logger(current_output_dir) # Initialize logger ASAP
        
        logging.info(f"--- Script Started ---")
        logging.info(f"Selected BIOPAC file: {biopac_file}")
        logging.info(f"Selected Label file: {label_file}")
        logging.info(f"Selected Initial Output Folder: {initial_output_folder}")
        logging.info(f"Preliminary Output Directory: {current_output_dir}")

        # --- 3. Process BIOPAC Data ---
        biopac_start_ms, biopac_end_ms, biopac_data = process_biopac(biopac_file)
        if not biopac_data: # Check if biopac_data is empty
            logging.error("BIOPAC data processing resulted in no data. Aborting.")
            messagebox.showerror("Error", f"BIOPAC data processing failed to yield any data from '{os.path.basename(biopac_file)}'. Check logs in '{current_output_dir}'.")
            return # Exit if no BIOPAC data

        # --- 4. Process Labels Data ---
        labels, metadata = process_labels(label_file)
        # `labels` can be an empty dictionary if no valid labels are found, which is acceptable.
        # `metadata` can be an empty dictionary if no header or relevant metadata found.

        # --- 5. Refine Output Directory Name with Metadata (if available) ---
        if metadata.get('subject') or metadata.get('session'): # Check if either is present
            subject = metadata.get('subject', 'unknown_subject').replace(" ", "_").replace("/", "-") # Sanitize
            session = metadata.get('session', 'unknown_session').replace(" ", "_").replace("/", "-") # Sanitize
            
            # Use a more descriptive name if subject/session available
            refined_name_parts = []
            if metadata.get('subject'):
                 refined_name_parts.append(subject)
            if metadata.get('session'):
                refined_name_parts.append(session)
            
            refined_folder_name = f"biopac_merge_{'_'.join(refined_name_parts)}_{timestamp_for_folder}"
            new_output_dir_path = os.path.join(initial_output_folder, refined_folder_name)

            if new_output_dir_path != current_output_dir:
                try:
                    # Before renaming, ensure the new path doesn't already exist UNLESS it's the same as current_output_dir
                    if os.path.exists(new_output_dir_path):
                        logging.warning(f"Target directory '{new_output_dir_path}' already exists. "
                                        f"Using the initially created timestamped directory: '{current_output_dir}' to avoid data loss.")
                    else:
                        os.rename(current_output_dir, new_output_dir_path)
                        logging.info(f"Output directory renamed from '{current_output_dir}' to '{new_output_dir_path}' based on metadata.")
                        current_output_dir = new_output_dir_path
                        # Re-initialize logger to the new directory
                        setup_logger(current_output_dir) 
                        logging.info(f"Logger re-initialized to new directory: {current_output_dir}")
                except OSError as e_rename:
                    logging.error(f"Could not rename output directory to '{new_output_dir_path}': {e_rename}. "
                                  f"Using original: '{current_output_dir}'")
        
        logging.info(f"Final Output Directory: {current_output_dir}")

        # --- 6. Determine Overall Time Range for Writing Data ---
        all_label_timestamps = list(labels.keys())
        # Effective start and end times for the output file, considering both data sources
        # Default to BIOPAC times, then adjust if labels extend beyond this range.
        effective_start_ms = biopac_start_ms
        effective_end_ms = biopac_end_ms

        if all_label_timestamps:
            min_label_ts = min(all_label_timestamps)
            max_label_ts = max(all_label_timestamps)
            # The output file should span the entirety of both datasets
            effective_start_ms = min(biopac_start_ms, min_label_ts)
            effective_end_ms = max(biopac_end_ms, max_label_ts)
        
        logging.info(f"Effective data range for output: {effective_start_ms}ms to {effective_end_ms}ms")

        # --- 7. Write Combined Data ---
        write_data_efficient(effective_start_ms, effective_end_ms, biopac_data, labels, current_output_dir, start_all)
        
        logging.info('--- Processing Completed Successfully ---')
        messagebox.showinfo("Success", f"Data processed successfully!\nOutput files are in: {current_output_dir}")

    except FileNotFoundError as e:
        # Log error even if logger wasn't fully set up (e.g. output_dir_for_log might be None)
        log_message = f"File not found error: {e}"
        if logging.getLogger().hasHandlers(): logging.error(log_message, exc_info=True)
        else: print(f"ERROR: {log_message}")
        messagebox.showerror("Error - File Not Found", f"Processing failed: {e}\nCheck logs in '{output_dir_for_log or 'your selected output folder'}' if available.")
    except NotADirectoryError as e:
        log_message = f"Directory not selected error: {e}"
        if logging.getLogger().hasHandlers(): logging.error(log_message, exc_info=True)
        else: print(f"ERROR: {log_message}")
        messagebox.showerror("Error - Directory Not Selected", f"Processing failed: {e}\nCheck logs in '{output_dir_for_log or 'your selected output folder'}' if available.")
    except ValueError as e: # Catch specific ValueErrors from processing functions
        log_message = f"Data processing or configuration error: {e}"
        if logging.getLogger().hasHandlers(): logging.error(log_message, exc_info=True)
        else: print(f"ERROR: {log_message}")
        messagebox.showerror("Error - Data/Config Problem", f"Processing failed: {e}\nCheck logs in '{current_output_dir or output_dir_for_log or 'your selected output folder'}' for details.")
    except Exception as e:
        log_message = f"A critical error occurred: {e}"
        if logging.getLogger().hasHandlers(): logging.critical(log_message, exc_info=True)
        else: print(f"CRITICAL ERROR: {log_message}")
        messagebox.showerror("Critical Error", f"An unexpected error occurred: {e}\nCheck logs in '{current_output_dir or output_dir_for_log or 'your selected output folder'}' for details.")
    finally:
        if logging.getLogger().hasHandlers(): # Check if logger was successfully initialized
            total_execution_time = time.time() - start_all
            logging.info(f"Total script execution time: {total_execution_time:.2f} seconds.")
            logging.info("--- Script Execution Finished ---")
        else: # Fallback print if logger failed very early
            total_execution_time = time.time() - start_all
            print(f"Total script execution time (logging might have failed): {total_execution_time:.2f} seconds.")
            print("--- Script Execution Finished (logging might have failed) ---")


if __name__ == "__main__":
    # Attempt to make Tkinter dialogs look more native on Windows (high DPI)
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1) 
    except ImportError:
        pass # If ctypes is not available (e.g. not on Windows)
    except AttributeError:
        pass # If shcore or SetProcessDpiAwareness is not available
    except Exception: # pylint: disable=bare-except
        pass # Catch any other unexpected errors silently
    main()

# import os
# import csv
# import logging
# import psutil
# from datetime import datetime
# import tkinter as tk
# from tkinter import filedialog, messagebox
# from tqdm import tqdm
# import bisect
# import time
# from typing import Dict, List, Tuple, Optional


# def find_nearest_timestamp(target: int, timestamps: List[int], window_ms: int = 1000) -> Optional[int]:
#     """Find first timestamp that is either exact match or next available within window."""
#     if not timestamps:
#         return None
        
#     idx = bisect.bisect_left(timestamps, target)
    
#     # Return exact match or next available timestamp within window
#     if idx < len(timestamps) and timestamps[idx] - target <= window_ms:
#         return timestamps[idx]
        
#     return None

# def write_data_efficient(start_time_ms: int, end_time_ms: int, 
#                         biopac_data: Dict, labels: Dict, 
#                         output_dir: str) -> None:
#     """Optimized data writing with label matching for BIOPAC device."""
#     os.makedirs(output_dir, exist_ok=True)
#     logging.info('Starting optimized data writing process')

#     # Pre-sort timestamps
#     biopac_timestamps = sorted(biopac_data.keys())
    
#     # Process labels and track assignments
#     label_assignments = []
#     biopac_label_matches = {}
    
#     for label_ts in sorted(labels.keys()):
#         label = labels[label_ts]
#         biopac_match = find_nearest_timestamp(label_ts, biopac_timestamps)
        
#         if biopac_match:
#             assignment = {
#                 'original_ts': label_ts,
#                 'label': label,
#                 'biopac_ts': biopac_match,
#                 'biopac_row': biopac_timestamps.index(biopac_match)+2
#             }
#             label_assignments.append(assignment)
#             biopac_label_matches[biopac_match] = label

#     # Headers
#     biopac_header = ['timestamp_ms', 'ECG', 'PPG', 'SKT', 'EDA', 'label']

#     # Write data efficiently
#     with open(os.path.join(output_dir, 'biopac_labels.csv'), 'w', newline='') as bf, \
#          open(os.path.join(output_dir, 'label_assignments.csv'), 'w', newline='') as lf:

#         writers = {
#             'biopac': csv.writer(bf),
#             'labels': csv.writer(lf)
#         }

#         # Write headers
#         writers['biopac'].writerow(biopac_header)
#         writers['labels'].writerow(['label', 'original_timestamp_ms', 'biopac_timestamp_ms', 
#                                   'biopac_row', 'time_diff_ms'])

#         # Write BIOPAC data with labels
#         for ts in biopac_timestamps:
#             values = biopac_data[ts]
#             label = biopac_label_matches.get(ts, '')
#             writers['biopac'].writerow([ts] + values + [label])

#         # Write label assignments
#         for assign in label_assignments:
#             time_diff = abs(assign['original_ts'] - assign['biopac_ts'])
#             writers['labels'].writerow([
#                 assign['label'],
#                 assign['original_ts'],
#                 assign['biopac_ts'],
#                 assign['biopac_row'],
#                 time_diff
#             ])

#     # Write performance metrics
#     diagnostic_data = {
#         'Processing Statistics': {
#             'Total Processing Time': f"{time.time() - start_all:.2f}s",
#             'Peak Memory Usage': f"{psutil.Process().memory_info().rss / (1024*1024):.2f}MB",
#             'BIOPAC Points': len(biopac_data),
#             'Labels Processed': len(labels),
#             'Labels Matched': len(label_assignments)
#         }
#     }

#     with open(os.path.join(output_dir, 'performance_metrics.txt'), 'w') as f:
#         for section, metrics in diagnostic_data.items():
#             f.write(f"\n{section}:\n")
#             f.write("="*50 + "\n")
#             for key, value in metrics.items():
#                 f.write(f"{key}: {value}\n")

# def process_labels(label_file, time_offset=0):
#     labels = {}
#     metadata = {}
#     logging.info('Starting label data processing.')
#     start_time = time.time()

#     with open(label_file, 'r') as f:
#         reader = csv.reader(f)
#         for idx, row in enumerate(reader):
#             try:
#                 timestamp_ms = int(float(row[0])) + time_offset
#                 label = row[1]
#                 labels[timestamp_ms] = label

#                 if idx == 0 and len(row) >= 6:
#                     metadata = {
#                         'device': row[2],
#                         'session': row[3],
#                         'subject': row[4],
#                         'trial': row[5]
#                     }
#             except (ValueError, IndexError):
#                 continue

#     processing_time = time.time() - start_time
#     logging.info(f'Finished label data processing. Time taken: {processing_time:.2f} seconds.')
#     logging.info(f'Labels processed: {len(labels)}')
#     return labels, metadata

# def process_biopac(biopac_file):
#     """
#     Process BIOPAC file with fixed sample rate and channel mappings.
#     File format:
#     - Header contains "Recording on: YYYY-MM-DD HH:MM:SS.fff"
#     - Sample rate in "X msec/sample" format
#     - Fixed channel positions: CH2 (ECG), CH3 (PPG), CH7 (SKT), CH16 (EDA)
#     """
#     biopac_data = {}
#     logging.info('Starting BIOPAC data processing.')
#     start_time = time.time()

#     with open(biopac_file, 'r') as f:
#         lines = f.readlines()

#     # Parse sample rate
#     sample_rate_line = next((line for line in lines if "msec/sample" in line), None)
#     if not sample_rate_line:
#         raise ValueError("Could not find sample rate (msec/sample) in file header")
    
#     try:
#         sample_rate_ms = float(sample_rate_line.split()[0])
#         logging.info(f"Detected sample rate: {sample_rate_ms} msec/sample")
#     except (ValueError, IndexError) as e:
#         raise ValueError(f"Failed to parse sample rate from line: {sample_rate_line}") from e

#     # Parse recording timestamp
#     recording_line = next((line for line in lines if line.startswith('Recording on:')), None)
#     if not recording_line:
#         raise ValueError("Could not find 'Recording on:' timestamp in file")

#     time_str = recording_line.split(': ')[1].strip()
#     try:
#         start_datetime = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
#     except ValueError:
#         start_datetime = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
#     start_time_ms = int(start_datetime.timestamp() * 1000)

#     # Find data section with fixed channel positions
#     data_start_idx = None
#     for idx, line in enumerate(lines):
#         if line.startswith('sec,CH'):
#             data_start_idx = idx
#             # Verify header format
#             headers = line.strip().split(',')
#             if not all(x in headers for x in ['CH2', 'CH3', 'CH7', 'CH16']):
#                 raise ValueError("Missing required channels in header")
#             break

#     if not data_start_idx:
#         raise ValueError("Could not find data section starting with 'sec,CH'")

#     # Fixed channel positions (0-based index, accounting for 'sec' column)
#     channel_positions = {
#         'ECG': 2,  # CH2
#         'PPG': 3,  # CH3
#         'SKT': 4,  # CH7
#         'EDA': 5   # CH16
#     }

#     # Skip the samples count line
#     data_start_idx += 2
    
#     # Process data using sample rate for timestamps
#     current_timestamp_ms = start_time_ms

#     for line in tqdm(lines[data_start_idx:], desc="Processing BIOPAC data", unit="line"):
#         try:
#             values = line.strip().split(',')
#             if not values[0] or values[0] == '':
#                 continue

#             # Extract values in standard order
#             extracted_values = [
#                 values[channel_positions['ECG']],  # ECG
#                 values[channel_positions['PPG']],  # PPG
#                 values[channel_positions['SKT']],  # SKT
#                 values[channel_positions['EDA']]   # EDA
#             ]
            
#             biopac_data[current_timestamp_ms] = extracted_values
#             current_timestamp_ms += int(sample_rate_ms)  # Increment by sample rate

#         except (ValueError, IndexError) as e:
#             logging.warning(f"Error processing line: {line.strip()}. Error: {str(e)}")
#             continue

#     total_duration_ms = (len(biopac_data) - 1) * int(sample_rate_ms)
#     expected_samples = total_duration_ms / sample_rate_ms + 1

#     logging.info(f'BIOPAC Processing Summary:')
#     logging.info(f'Start time: {start_datetime}')
#     logging.info(f'Sample rate: {sample_rate_ms} ms ({1000/sample_rate_ms:.2f} Hz)')
#     logging.info(f'Samples processed: {len(biopac_data)}')
#     logging.info(f'Expected samples: {expected_samples}')
#     logging.info(f'Total duration: {total_duration_ms/1000:.2f} seconds')
#     logging.info(f'Processing time: {time.time() - start_time:.2f} seconds')
    
#     return start_time_ms, current_timestamp_ms - int(sample_rate_ms), biopac_data


# def setup_logger(output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     logging.basicConfig(
#         filename=os.path.join(output_dir, 'performance.log'),
#         filemode='w',
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         level=logging.INFO
#     )
#     logging.info('Logger initialized.')

# def select_files():
#     """Select input and output files/folders."""
#     root = tk.Tk()
#     root.withdraw()

#     biopac_file = filedialog.askopenfilename(
#         title="Select BIOPAC file",
#         filetypes=[("Text files", "*.txt")]
#     )
#     if not biopac_file:
#         raise FileNotFoundError("BIOPAC file not selected.")

#     label_file = filedialog.askopenfilename(
#         title="Select label file",
#         filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")]
#     )
#     if not label_file:
#         raise FileNotFoundError("Label file not selected.")
        
#     output_folder = filedialog.askdirectory(
#         title="Select output folder for processed data"
#     )
#     if not output_folder:
#         raise NotADirectoryError("Output folder not selected.")

#     return biopac_file, label_file, output_folder

# def main():
#     try:
#         global start_all
#         start_all = time.time()
#         biopac_file, label_file, output_folder = select_files()
        
#         # Create session-specific output directory
#         timestamp = int(time.time())
#         output_dir = os.path.join(output_folder, f"biopac_merge_{timestamp}")
#         # setup_logger(output_dir)
        
#         start_time_ms, biopac_end_ms, biopac_data = process_biopac(biopac_file)
#         labels, metadata = process_labels(label_file)

#         # Update output directory with metadata
#         if metadata:
#             output_dir = os.path.join(output_folder,
#                 f"biopac_merge_{metadata.get('subject', 'unknown')}_{metadata.get('session', 'unknown')}")
#             os.makedirs(output_dir, exist_ok=True)

#         end_time_ms = max(
#             biopac_end_ms,
#             max(labels.keys()) if labels else start_time_ms
#         )

#         write_data_efficient(start_time_ms, end_time_ms, biopac_data, labels, output_dir)
        
#         logging.info('Processing completed successfully')
#         messagebox.showinfo("Success", f"Data processed successfully!\nOutput directory: {output_dir}")

#     except Exception as e:
#         logging.exception('Fatal error occurred')
#         messagebox.showerror("Error", f"Processing failed:\n{str(e)}")

# if __name__ == "__main__":
#     main()