# MINDER_mDAQ_BIOPAC_label_merge
## mDAQ and BIOPAC Data Merge Scripts

These python scripts merge physiological data collected from BIOPAC and mDAQ devices by synchronizing them based on timestamps. They help prepare your data for analysis by:
- Combining data from both devices into aligned timelines
- Matching event labels with the corresponding physiological measurements
- Handling different sampling rates (BIOPAC: 250/1000 Hz, mDAQ: 128 Hz and 1 Hz)
- Generating analysis-ready CSV files

## Two Approaches Available

### 1. Millisecond-by-Millisecond (merge_mDAQ_BIOPAC_approach1ms.py)
- Creates a complete timeline with every millisecond between start and end times
- Places each data point at its exact timestamp
- Leaves gaps between samples to maintain original sampling rates
- Best when you need precise temporal relationships between signals
- Generates an additional merged_data.csv with combined device data

### 2. Nearest Timestamp (merge_mDAQ_BIOPAC_nearest_ms.py)
- Matches data points and labels to their nearest available timestamp
- Uses a 1-second window for matching
- More efficient with memory and processing time
- Better for large datasets where exact millisecond precision isn't critical
- Provides statistics about timing matches

## Prerequisites

1. Python 3.x with packages:
```bash
pip install tqdm psutil
```

2. Required files:
- BIOPAC data file (.txt)
  - Contains: ECG, PPG, SKT, EDA
  - Has header with recording time
- mDAQ folder with .csv files
  - Contains: ECG, EDA, PPG (IR/Red), accelerometer, gyroscope, temperature, humidity
  - Files should be numbered (e.g., 1.csv, 2.csv)
- Label file (.csv)
  - Required columns: timestamp, label
  - Optional: device, session, subject, trial

## Usage

1. Run either script:
```bash
python merge_mDAQ_BIOPAC_approach1ms.py
# OR
python merge_mDAQ_BIOPAC_nearest_ms.py
```

2. Use the pop-up windows to select:
- BIOPAC file (.txt)
- Label file (.csv)
- mDAQ folder
- Output folder

3. Wait for processing to complete
- Progress bars will show status
- Success message will display output location

## Output Files

Both scripts create:
1. `biopac_labels.csv`
   - BIOPAC data with matched labels
   - Columns: timestamp_ms, ECG, PPG, SKT, EDA, label

2. `mdaq_labels.csv`
   - mDAQ data with matched labels
   - Columns: timestamp_ms, ecg, eda, ir, red, acc_x/y/z, gyr_x/y/z, environmental data, label

3. `label_assignments.csv`
   - Shows how labels were matched to data points
   - Includes row numbers for easy reference

4. `performance_metrics.txt`
   - Processing time and memory usage
   - Data point counts
   - Label matching statistics

Additional output for Approach 1:
- `merged_data.csv`: Combined data from both devices for each millisecond

## Common Issues

1. File Selection Errors
   - Ensure BIOPAC file is .txt format
   - mDAQ folder should contain numbered .csv files
   - Label file should be .csv format

2. Memory Issues
   - Try Approach 2 for large datasets
   - Close other memory-intensive applications

3. Data Alignment
   - Verify devices were time-synchronized during data collection
   - Check label timestamps fall within data collection period

## Need Help?

For issues or questions:
- Check the performance.log file for error details
- Include error messages when requesting support
- Specify which approach you're using