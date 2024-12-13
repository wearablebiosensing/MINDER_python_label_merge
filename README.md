# MINDER_MDAQ_BIOPAC_LABEL_MERGE

A Python-based toolset for processing and visualizing physiological data from BIOPAC and mDAQ devices. This repository contains three main components:

1. BIOPAC Label Merger
2. BIOPAC-mDAQ Data Synchronizer
3. Interactive Data Visualizer

## Repository Structure

```
MINDER_MDAQ_BIOPAC_LABEL_MERGE/
├── .gitattributes
├── BIOPAC_label_merge.py
├── merge_mDAQ_BIOPAC_nearest_ms.py
├── mDAQ_BIOPAC_data_visualizer.py
├── LICENSE
└── README.md
```

## Components

### 1. BIOPAC_label_merge.py

A Python script for merging BIOPAC physiological data with event labels.

**Features:**
- Processes BIOPAC files containing ECG, PPG, SKT, and EDA data
- Matches timestamps with event labels using nearest-neighbor approach
- Generates labeled output with detailed matching information
- Memory-efficient processing for large datasets

### 2. merge_mDAQ_BIOPAC_nearest_ms.py

Combines data from both BIOPAC and mDAQ devices with event labels.

**Features:**
- Synchronizes multi-device data based on timestamps
- Handles different sampling rates between devices
- Processes multiple data channels:
  - BIOPAC: ECG, PPG, SKT, EDA
  - mDAQ: ECG, EDA, IR/Red PPG, accelerometer, gyroscope, temperature, humidity
- Generates separate labeled files for each device

### 3. mDAQ_BIOPAC_data_visualizer.py

A PyQt5-based GUI application for visualizing synchronized physiological data.

**Features:**
- Real-time data visualization
- Customizable display options:
  - Adjustable time window
  - Channel selection
  - Color schemes
  - Grid and background settings
- Label visualization with configurable:
  - Marker styles
  - Text positions
  - Line types
- Linked timeline across all channels

## Prerequisites

- Python 3.8+
- Required Python packages:
  ```
  pandas
  numpy
  PyQt5
  pyqtgraph
  tqdm
  ```

## File Requirements

### BIOPAC Files (.txt)
- Must contain header with recording time
- Required channels: ECG, PPG, SKT, EDA
- Sample rate information in msec/sample format

### mDAQ Files (.csv)
- Numbered CSV files (e.g., 1.csv, 2.csv)
- Contains physiological and environmental data
- Standard mDAQ data format with ISI values

### Label Files (.csv)
Required columns:
- timestamp_ms: Event timestamps in milliseconds
- label: Event markers/labels

Optional metadata columns:
- device
- session
- subject
- trial

## Usage

### BIOPAC Label Merger
```python
python BIOPAC_label_merge.py
```
1. Select BIOPAC .txt file
2. Select label .csv file
3. Choose output directory

### BIOPAC-mDAQ Synchronizer
```python
python merge_mDAQ_BIOPAC_nearest_ms.py
```
1. Select BIOPAC .txt file
2. Select mDAQ folder containing .csv files
3. Select label .csv file
4. Choose output directory

### Data Visualizer
```python
python mDAQ_BIOPAC_data_visualizer.py
```
1. Launch the GUI application
2. Load BIOPAC and/or mDAQ files using the interface
3. Adjust visualization settings as needed

## Output Files

The processing scripts generate:

1. `biopac_labels.csv`
   - BIOPAC data with matched labels
   - Columns: timestamp_ms, ECG, PPG, SKT, EDA, label

2. `mdaq_labels.csv` (if using synchronizer)
   - mDAQ data with matched labels
   - All mDAQ channels with matched labels

3. `label_assignments.csv`
   - Label matching details
   - Includes original timestamps and matching information

4. `performance_metrics.txt`
   - Processing statistics
   - Data points processed
   - Memory usage
   - Processing time

## Common Issues

### Data Processing
- Ensure consistent timestamp formats across files
- Verify data collection period overlap between devices
- Check for missing or corrupted data in input files

### Visualization
- Large files may require more memory
- Adjust time window for smoother performance
- Use channel selection to focus on relevant data

## Support

For technical issues:
- Check the performance_metrics.txt file
- Review console output for error messages
- Include sample data when reporting problems