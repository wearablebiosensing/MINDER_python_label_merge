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
                        output_dir: str) -> None:
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
                'biopac_row': biopac_timestamps.index(biopac_match)+2
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
            'Total Processing Time': f"{time.time() - start_all:.2f}s",
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

def process_labels(label_file, time_offset=0):
    labels = {}
    metadata = {}
    logging.info('Starting label data processing.')
    start_time = time.time()

    with open(label_file, 'r') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            try:
                timestamp_ms = int(float(row[0])) + time_offset
                label = row[1]
                labels[timestamp_ms] = label

                if idx == 0 and len(row) >= 6:
                    metadata = {
                        'device': row[2],
                        'session': row[3],
                        'subject': row[4],
                        'trial': row[5]
                    }
            except (ValueError, IndexError):
                continue

    processing_time = time.time() - start_time
    logging.info(f'Finished label data processing. Time taken: {processing_time:.2f} seconds.')
    logging.info(f'Labels processed: {len(labels)}')
    return labels, metadata

def process_biopac(biopac_file):
    """
    Process BIOPAC file with fixed sample rate and channel mappings.
    File format:
    - Header contains "Recording on: YYYY-MM-DD HH:MM:SS.fff"
    - Sample rate in "X msec/sample" format
    - Fixed channel positions: CH2 (ECG), CH3 (PPG), CH7 (SKT), CH16 (EDA)
    """
    biopac_data = {}
    logging.info('Starting BIOPAC data processing.')
    start_time = time.time()

    with open(biopac_file, 'r') as f:
        lines = f.readlines()

    # Parse sample rate
    sample_rate_line = next((line for line in lines if "msec/sample" in line), None)
    if not sample_rate_line:
        raise ValueError("Could not find sample rate (msec/sample) in file header")
    
    try:
        sample_rate_ms = float(sample_rate_line.split()[0])
        logging.info(f"Detected sample rate: {sample_rate_ms} msec/sample")
    except (ValueError, IndexError) as e:
        raise ValueError(f"Failed to parse sample rate from line: {sample_rate_line}") from e

    # Parse recording timestamp
    recording_line = next((line for line in lines if line.startswith('Recording on:')), None)
    if not recording_line:
        raise ValueError("Could not find 'Recording on:' timestamp in file")

    time_str = recording_line.split(': ')[1].strip()
    try:
        start_datetime = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        start_datetime = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    start_time_ms = int(start_datetime.timestamp() * 1000)

    # Find data section with fixed channel positions
    data_start_idx = None
    for idx, line in enumerate(lines):
        if line.startswith('sec,CH'):
            data_start_idx = idx
            # Verify header format
            headers = line.strip().split(',')
            if not all(x in headers for x in ['CH2', 'CH3', 'CH7', 'CH16']):
                raise ValueError("Missing required channels in header")
            break

    if not data_start_idx:
        raise ValueError("Could not find data section starting with 'sec,CH'")

    # Fixed channel positions (0-based index, accounting for 'sec' column)
    channel_positions = {
        'ECG': 2,  # CH2
        'PPG': 3,  # CH3
        'SKT': 4,  # CH7
        'EDA': 5   # CH16
    }

    # Skip the samples count line
    data_start_idx += 2
    
    # Process data using sample rate for timestamps
    current_timestamp_ms = start_time_ms

    for line in tqdm(lines[data_start_idx:], desc="Processing BIOPAC data", unit="line"):
        try:
            values = line.strip().split(',')
            if not values[0] or values[0] == '':
                continue

            # Extract values in standard order
            extracted_values = [
                values[channel_positions['ECG']],  # ECG
                values[channel_positions['PPG']],  # PPG
                values[channel_positions['SKT']],  # SKT
                values[channel_positions['EDA']]   # EDA
            ]
            
            biopac_data[current_timestamp_ms] = extracted_values
            current_timestamp_ms += int(sample_rate_ms)  # Increment by sample rate

        except (ValueError, IndexError) as e:
            logging.warning(f"Error processing line: {line.strip()}. Error: {str(e)}")
            continue

    total_duration_ms = (len(biopac_data) - 1) * int(sample_rate_ms)
    expected_samples = total_duration_ms / sample_rate_ms + 1

    logging.info(f'BIOPAC Processing Summary:')
    logging.info(f'Start time: {start_datetime}')
    logging.info(f'Sample rate: {sample_rate_ms} ms ({1000/sample_rate_ms:.2f} Hz)')
    logging.info(f'Samples processed: {len(biopac_data)}')
    logging.info(f'Expected samples: {expected_samples}')
    logging.info(f'Total duration: {total_duration_ms/1000:.2f} seconds')
    logging.info(f'Processing time: {time.time() - start_time:.2f} seconds')
    
    return start_time_ms, current_timestamp_ms - int(sample_rate_ms), biopac_data


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(output_dir, 'performance.log'),
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.info('Logger initialized.')

def select_files():
    """Select input and output files/folders."""
    root = tk.Tk()
    root.withdraw()

    biopac_file = filedialog.askopenfilename(
        title="Select BIOPAC file",
        filetypes=[("Text files", "*.txt")]
    )
    if not biopac_file:
        raise FileNotFoundError("BIOPAC file not selected.")

    label_file = filedialog.askopenfilename(
        title="Select label file",
        filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")]
    )
    if not label_file:
        raise FileNotFoundError("Label file not selected.")
        
    output_folder = filedialog.askdirectory(
        title="Select output folder for processed data"
    )
    if not output_folder:
        raise NotADirectoryError("Output folder not selected.")

    return biopac_file, label_file, output_folder

def main():
    try:
        global start_all
        start_all = time.time()
        biopac_file, label_file, output_folder = select_files()
        
        # Create session-specific output directory
        timestamp = int(time.time())
        output_dir = os.path.join(output_folder, f"biopac_merge_{timestamp}")
        # setup_logger(output_dir)
        
        start_time_ms, biopac_end_ms, biopac_data = process_biopac(biopac_file)
        labels, metadata = process_labels(label_file)

        # Update output directory with metadata
        if metadata:
            output_dir = os.path.join(output_folder,
                f"biopac_merge_{metadata.get('subject', 'unknown')}_{metadata.get('session', 'unknown')}")
            os.makedirs(output_dir, exist_ok=True)

        end_time_ms = max(
            biopac_end_ms,
            max(labels.keys()) if labels else start_time_ms
        )

        write_data_efficient(start_time_ms, end_time_ms, biopac_data, labels, output_dir)
        
        logging.info('Processing completed successfully')
        messagebox.showinfo("Success", f"Data processed successfully!\nOutput directory: {output_dir}")

    except Exception as e:
        logging.exception('Fatal error occurred')
        messagebox.showerror("Error", f"Processing failed:\n{str(e)}")

if __name__ == "__main__":
    main()