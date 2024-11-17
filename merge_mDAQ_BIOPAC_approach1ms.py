import os
import csv
import logging
import psutil
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from tqdm import tqdm
import time

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

def select_files():
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

    return biopac_file, mdaq_folder, label_file

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

def write_data_efficient(start_time_ms, end_time_ms, biopac_data, mdaq_data, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.info('Starting data writing process.')

    # Headers
    biopac_header = ['timestamp_ms', 'ECG', 'PPG', 'SKT', 'EDA', 'label']
    mdaq_header = ['timestamp_ms', 'ecg', 'eda', 'ir', 'red', 'acc_x', 'acc_y', 'acc_z',
                   'gyr_x', 'gyr_y', 'gyr_z', 'batt%', 'relative_humidity', 'ambient_temp', 'body_temp', 'label']


    merged_header = ['timestamp_ms'] + biopac_header[1:-1] + mdaq_header[1:]

    # Empty values
    biopac_empty = [''] * 4
    mdaq_empty = [''] * 14  # 10 fast + 4 slow channels

    # Buffer size for efficient writing
    buffer_size = 100000
    biopac_buffer = []
    mdaq_buffer = []
    merged_buffer = []

    biopac_count = 0
    mdaq_count = 0
    total_count = 0

    # Initialize label assignments list
    label_assignments = []

    with open(os.path.join(output_dir, 'biopac_labels.csv'), 'w', newline='') as bf, \
         open(os.path.join(output_dir, 'mdaq_labels.csv'), 'w', newline='') as mf, \
         open(os.path.join(output_dir, 'merged_data.csv'), 'w', newline='') as mergf:

        writers = {
            'biopac': csv.writer(bf),
            'mdaq': csv.writer(mf),
            'merged': csv.writer(mergf)
        }

        # Write headers
        writers['biopac'].writerow(biopac_header)
        writers['mdaq'].writerow(mdaq_header)
        writers['merged'].writerow(merged_header)

        # Process each millisecond
        for ms_idx, ms in enumerate(tqdm(range(start_time_ms, end_time_ms + 1), desc="Writing data", unit="timestamp")):
            biopac_values = biopac_data.get(ms, biopac_empty)
            mdaq_values = mdaq_data.get(ms, mdaq_empty)
            label = labels.get(ms, '')

            # Record label assignments
            if label:
                label_assignments.append({
                    'timestamp_ms': ms,
                    'row': ms_idx + 2,  # +2 accounting for header and zero-based index
                    'label': label
                })

            biopac_buffer.append([ms] + biopac_values + [label])
            mdaq_buffer.append([ms] + mdaq_values + [label])
            merged_buffer.append([ms] + biopac_values + mdaq_values + [label])

            if biopac_values != biopac_empty:
                biopac_count += 1
            if mdaq_values != mdaq_empty:
                mdaq_count += 1
            total_count += 1

            # Write when buffer is full
            if len(biopac_buffer) >= buffer_size:
                writers['biopac'].writerows(biopac_buffer)
                writers['mdaq'].writerows(mdaq_buffer)
                writers['merged'].writerows(merged_buffer)
                biopac_buffer = []
                mdaq_buffer = []
                merged_buffer = []

        # Write remaining buffer
        if biopac_buffer:
            writers['biopac'].writerows(biopac_buffer)
            writers['mdaq'].writerows(mdaq_buffer)
            writers['merged'].writerows(merged_buffer)

    # Output label assignments to a CSV file
    with open(os.path.join(output_dir, 'label_assignments.csv'), 'w', newline='') as lafile:
        la_writer = csv.writer(lafile)
        la_writer.writerow(['label', 'timestamp_ms', 'biopac_row', 'mdaq_row'])

        for assignment in label_assignments:
            timestamp = assignment['timestamp_ms']
            label = assignment['label']
            row_number = assignment['row']
            la_writer.writerow([label, timestamp, row_number, row_number])

    logging.info('Data writing process completed.')
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_memory_mb = mem_info.rss / (1024 * 1024)

    performance_metrics = {
        "Total Processing Time (s)": f"{time.time() - start_all:.2f}",
        "Peak Memory Usage (MB)": f"{rss_memory_mb:.2f}",
        "BIOPAC Data Points": biopac_count,
        "mDAQ Data Points": mdaq_count,
        "Total Data Points": total_count,
        "Labels Processed": len(labels),
    }

    with open(os.path.join(output_dir, 'performance_metrics.txt'), 'w') as perf_file:
        for key, value in performance_metrics.items():
            perf_file.write(f"{key}: {value}\n")

def main():
    try:
        global start_all
        start_all = time.time()
        
        # Get input/output paths
        biopac_file, mdaq_folder, label_file, output_folder = select_folders()
        
        # Create session-specific output directory
        timestamp = int(time.time())
        output_dir = os.path.join(output_folder, f"processed_data_{timestamp}")
        setup_logger(output_dir)
        
        # Process data
        start_time_ms, biopac_end_ms, biopac_data = process_biopac(biopac_file)
        mdaq_data, mdaq_end_ms = process_mdaq(mdaq_folder, start_time_ms)
        labels, metadata = process_labels(label_file)

        # Update output directory with metadata if available
        if metadata:
            output_dir = os.path.join(output_folder, 
                f"processed_{metadata.get('subject', 'unknown')}_{metadata.get('session', 'unknown')}_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)

        end_time_ms = max(
            biopac_end_ms,
            mdaq_end_ms,
            max(labels.keys()) if labels else start_time_ms
        )

        # Write data
        write_data_efficient(start_time_ms, end_time_ms, biopac_data, mdaq_data, labels, output_dir)
        
        logging.info('Processing completed successfully')
        messagebox.showinfo("Success", 
            f"Data processed successfully!\nOutput directory:\n{output_dir}")

    except Exception as e:
        logging.exception('Fatal error occurred')
        messagebox.showerror("Error", f"Processing failed:\n{str(e)}")

if __name__ == "__main__":
    main()