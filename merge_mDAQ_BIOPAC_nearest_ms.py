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
import statistics
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
                        biopac_data: Dict, mdaq_data: Dict, 
                        labels: Dict, output_dir: str) -> None:
    """Optimized data writing with independent label matching for each device."""
    os.makedirs(output_dir, exist_ok=True)
    logging.info('Starting optimized data writing process')

    # Pre-sort timestamps
    biopac_timestamps = sorted(biopac_data.keys())
    mdaq_timestamps = sorted(mdaq_data.keys())
    
    # Process labels and track assignments
    label_assignments = []
    biopac_label_matches = {}
    mdaq_label_matches = {}
    
    for label_ts in sorted(labels.keys()):
        label = labels[label_ts]
        biopac_match = find_nearest_timestamp(label_ts, biopac_timestamps)
        mdaq_match = find_nearest_timestamp(label_ts, mdaq_timestamps)
        
        if biopac_match or mdaq_match:
            assignment = {
                'original_ts': label_ts,
                'label': label,
                'biopac_ts': biopac_match,
                'mdaq_ts': mdaq_match,
                'biopac_row': biopac_timestamps.index(biopac_match)+2 if biopac_match else None,
                'mdaq_row': mdaq_timestamps.index(mdaq_match)+2 if mdaq_match else None
            }
            label_assignments.append(assignment)
            
            if biopac_match:
                biopac_label_matches[biopac_match] = label
            if mdaq_match:
                mdaq_label_matches[mdaq_match] = label

    # Headers
    biopac_header = ['timestamp_ms', 'ECG', 'PPG', 'SKT', 'EDA', 'label']
    mdaq_header = ['timestamp_ms', 'ecg', 'eda', 'ir', 'red', 'acc_x', 'acc_y', 'acc_z',
                   'gyr_x', 'gyr_y', 'gyr_z', 'batt%', 'relative_humidity', 'ambient_temp', 'body_temp', 'label']

    # Write data efficiently
    with open(os.path.join(output_dir, 'biopac_labels.csv'), 'w', newline='') as bf, \
         open(os.path.join(output_dir, 'mdaq_labels.csv'), 'w', newline='') as mf, \
         open(os.path.join(output_dir, 'label_assignments.csv'), 'w', newline='') as lf:

        writers = {
            'biopac': csv.writer(bf),
            'mdaq': csv.writer(mf),
            'labels': csv.writer(lf)
        }

        # Write headers
        writers['biopac'].writerow(biopac_header)
        writers['mdaq'].writerow(mdaq_header)
        writers['labels'].writerow(['label', 'original_timestamp_ms', 'biopac_timestamp_ms', 
                                  'mdaq_timestamp_ms', 'biopac_row', 'mdaq_row', 'time_diff_ms'])

        # Write BIOPAC data with labels
        for ts in biopac_timestamps:
            values = biopac_data[ts]
            label = biopac_label_matches.get(ts, '')
            writers['biopac'].writerow([ts] + values + [label])

        # Write mDAQ data with labels
        for ts in mdaq_timestamps:
            values = mdaq_data[ts]
            label = mdaq_label_matches.get(ts, '')
            writers['mdaq'].writerow([ts] + values + [label])

        # Write label assignments
        for assign in label_assignments:
            matched_ts = assign['biopac_ts'] or assign['mdaq_ts']
            time_diff = abs(assign['original_ts'] - matched_ts) if matched_ts else None
            writers['labels'].writerow([
                assign['label'],
                assign['original_ts'],
                assign['biopac_ts'] or '',
                assign['mdaq_ts'] or '',
                assign['biopac_row'] or '',
                assign['mdaq_row'] or '',
                time_diff or ''
            ])

    # Write performance metrics
    diagnostic_data = {
        'Processing Statistics': {
            'Total Processing Time': f"{time.time() - start_all:.2f}s",
            'Peak Memory Usage': f"{psutil.Process().memory_info().rss / (1024*1024):.2f}MB",
            'BIOPAC Points': len(biopac_data),
            'mDAQ Points': len(mdaq_data),
            'Labels Processed': len(labels),
            'Labels Matched': len(label_assignments),
            'BIOPAC Labels': len(biopac_label_matches),
            'mDAQ Labels': len(mdaq_label_matches)
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
    biopac_data = {}
    logging.info('Starting BIOPAC data processing.')
    start_time = time.time()

    with open(biopac_file, 'r') as f:
        lines = f.readlines()

    time_str = lines[2].split(': ')[1].strip()
    try:
        start_datetime = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        start_datetime = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    start_time_ms = int(start_datetime.timestamp() * 1000)

    # Process data
    for line in tqdm(lines[15:], desc="Processing BIOPAC data", unit="line"):
        try:
            values = line.strip().split(',')
            elapsed_s = float(values[0])
            timestamp_ms = start_time_ms + int(elapsed_s * 1000)
            biopac_data[timestamp_ms] = values[1:5]
        except (ValueError, IndexError):
            continue

    processing_time = time.time() - start_time
    logging.info(f'Finished BIOPAC data processing. Time taken: {processing_time:.2f} seconds.')
    logging.info(f'BIOPAC samples: {len(biopac_data)}')
    return start_time_ms, max(biopac_data.keys()), biopac_data

def process_mdaq(mdaq_folder, start_time_ms):
    mdaq_data = {}
    isi_accumulator = 0
    logging.info('Starting mDAQ data processing.')
    start_time = time.time()

    files = [f for f in os.listdir(mdaq_folder)
             if f.endswith('.csv') and f[:-4].isdigit()]
    sorted_files = sorted(files, key=lambda x: int(x[:-4]))

    for filename in tqdm(sorted_files, desc="Processing mDAQ files", unit="file"):
        file_path = os.path.join(mdaq_folder, filename)
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header

            for row in reader:
                try:
                    isi_us = int(float(row[11]))
                    isi_accumulator += isi_us
                    timestamp_ms = start_time_ms + (isi_accumulator // 1000)

                    values = [
                        row[0],  # ECG
                        str(335544320 / float(row[1])) if float(row[1]) != 0 else '',  # EDA
                        row[2],  # IR
                        row[3],  # Red
                        row[4],  # acc_x
                        row[5],  # acc_y
                        row[6],  # acc_z
                        row[7],  # gyr_x
                        row[8],  # gyr_y
                        row[9]   # gyr_z
                    ]

                    values.extend(row[12:16] if len(row) >= 16 else [''] * 4)

                    mdaq_data[timestamp_ms] = values

                except (ValueError, IndexError, ZeroDivisionError):
                    continue

    processing_time = time.time() - start_time
    logging.info(f'Finished mDAQ data processing. Time taken: {processing_time:.2f} seconds.')
    logging.info(f'mDAQ samples: {len(mdaq_data)}')
    return mdaq_data, max(mdaq_data.keys()) if mdaq_data else start_time_ms


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(output_dir, 'performance.log'),
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.info('Logger initialized.')


def select_folders():
    """Select input and output folders/files."""
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

    mdaq_folder = filedialog.askdirectory(
        title="Select mDAQ folder containing CSV files"
    )
    if not mdaq_folder:
        raise NotADirectoryError("mDAQ folder not selected.")
        
    output_folder = filedialog.askdirectory(
        title="Select output folder for processed data"
    )
    if not output_folder:
        raise NotADirectoryError("Output folder not selected.")

    return biopac_file, mdaq_folder, label_file, output_folder


def main():
    try:
        global start_all
        start_all = time.time()
        biopac_file, mdaq_folder, label_file, output_folder = select_folders()
        # Create session-specific output directory
        timestamp = int(time.time())
        output_dir = os.path.join(output_folder, f"approach2_{timestamp}")
        setup_logger(output_dir)

        start_time_ms, biopac_end_ms, biopac_data = process_biopac(biopac_file)
        mdaq_data, mdaq_end_ms = process_mdaq(mdaq_folder, start_time_ms)
        labels, metadata = process_labels(label_file)

        # Update output directory with metadata
        if metadata:
            output_dir = os.path.join(output_folder,
                f"approach2_{metadata.get('subject', 'unknown')}_{metadata.get('session', 'unknown')}_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)


        end_time_ms = max(
            biopac_end_ms,
            mdaq_end_ms,
            max(labels.keys()) if labels else start_time_ms
        )

        write_data_efficient(start_time_ms, end_time_ms, biopac_data, mdaq_data, labels, output_dir)
        
        logging.info('Processing completed successfully')
        messagebox.showinfo("Success", f"Data processed successfully!\nOutput directory: {output_dir}")

    except Exception as e:
        logging.exception('Fatal error occurred')
        messagebox.showerror("Error", f"Processing failed:\n{str(e)}")

if __name__ == "__main__":
    main()