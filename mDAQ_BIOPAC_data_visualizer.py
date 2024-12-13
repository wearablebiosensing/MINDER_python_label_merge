import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel
import pyqtgraph as pg

class TimeSeriesViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Physiological Data Viewer')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create file picker buttons
        self.mdaq_button = QPushButton('Select mDAQ File')
        self.biopac_button = QPushButton('Select BIOPAC File')
        self.mdaq_button.clicked.connect(lambda: self.load_file('mdaq'))
        self.biopac_button.clicked.connect(lambda: self.load_file('biopac'))
        
        # Add file labels
        self.mdaq_label = QLabel('No mDAQ file selected')
        self.biopac_label = QLabel('No BIOPAC file selected')
        
        # Create plot widget
        self.plot_widget = pg.GraphicsLayoutWidget()
        
        # Add widgets to layout
        layout.addWidget(self.mdaq_button)
        layout.addWidget(self.mdaq_label)
        layout.addWidget(self.biopac_button)
        layout.addWidget(self.biopac_label)
        layout.addWidget(self.plot_widget)
        
        # Initialize data storage
        self.mdaq_data = None
        self.biopac_data = None
        
        # Create plots
        self.setup_plots()
        
    def setup_plots(self):
        # Create plots with shared x-axis
        self.plots = {}
        
        # mDAQ plots
        self.plots['mdaq_ecg'] = self.plot_widget.addPlot(row=0, col=0, title='mDAQ ECG')
        self.plots['mdaq_eda'] = self.plot_widget.addPlot(row=1, col=0, title='mDAQ EDA')
        self.plots['mdaq_ppg'] = self.plot_widget.addPlot(row=2, col=0, title='mDAQ PPG (IR)')
        
        # BIOPAC plots
        self.plots['biopac_ecg'] = self.plot_widget.addPlot(row=0, col=1, title='BIOPAC ECG')
        self.plots['biopac_eda'] = self.plot_widget.addPlot(row=1, col=1, title='BIOPAC EDA')
        self.plots['biopac_ppg'] = self.plot_widget.addPlot(row=2, col=1, title='BIOPAC PPG')
        
        # Link x-axes for synchronized zooming/panning
        for plot in self.plots.values():
            plot.setXLink(self.plots['mdaq_ecg'])
            plot.showGrid(x=True, y=True)
            plot.addLegend()
    
    def load_file(self, file_type):
        filename, _ = QFileDialog.getOpenFileName(self, f'Select {file_type.upper()} File', '', 'CSV Files (*.csv)')
        
        if filename:
            if file_type == 'mdaq':
                self.mdaq_label.setText(f'mDAQ file: {filename}')
                self.load_mdaq_data(filename)
            else:
                self.biopac_label.setText(f'BIOPAC file: {filename}')
                self.load_biopac_data(filename)
    
    def load_mdaq_data(self, filename):
        # Load data efficiently using specified dtypes
        df = pd.read_csv(filename, dtype={
            'timestamp_ms': np.int64,
            'ecg': np.float32,
            'eda': np.float32,
            'ir': np.float32,
            'label': str
        })
        
        # Convert timestamp to relative time in seconds
        t0 = df['timestamp_ms'].iloc[0]
        time = (df['timestamp_ms'] - t0) / 1000.0
        
        # Plot data
        self.plots['mdaq_ecg'].clear()
        self.plots['mdaq_eda'].clear()
        self.plots['mdaq_ppg'].clear()
        
        self.plots['mdaq_ecg'].plot(time, df['ecg'], pen='b', name='ECG')
        self.plots['mdaq_eda'].plot(time, df['eda'], pen='r', name='EDA')
        self.plots['mdaq_ppg'].plot(time, df['ir'], pen='g', name='IR (PPG)')
        
        # Add labels as markers if they exist
        if 'label' in df.columns:
            label_points = df[df['label'].notna()]
            if not label_points.empty:
                label_time = (label_points['timestamp_ms'] - t0) / 1000.0
                for plot in [self.plots['mdaq_ecg'], self.plots['mdaq_eda'], self.plots['mdaq_ppg']]:
                    plot.plot(label_time, np.zeros_like(label_time), 
                            pen=None, symbol='o', symbolBrush='r', 
                            symbolSize=10, name='Labels')
    
    def load_biopac_data(self, filename):
        # Load data efficiently using specified dtypes
        df = pd.read_csv(filename, dtype={
            'timestamp_ms': np.int64,
            'ECG': np.float32,
            'EDA': np.float32,
            'PPG': np.float32,
            'label': str
        })
        
        # Convert timestamp to relative time in seconds
        t0 = df['timestamp_ms'].iloc[0]
        time = (df['timestamp_ms'] - t0) / 1000.0
        
        # Plot data
        self.plots['biopac_ecg'].clear()
        self.plots['biopac_eda'].clear()
        self.plots['biopac_ppg'].clear()
        
        self.plots['biopac_ecg'].plot(time, df['ECG'], pen='b', name='ECG')
        self.plots['biopac_eda'].plot(time, df['EDA'], pen='r', name='EDA')
        self.plots['biopac_ppg'].plot(time, df['PPG'], pen='g', name='PPG')
        
        # Add labels as markers if they exist
        if 'label' in df.columns:
            label_points = df[df['label'].notna()]
            if not label_points.empty:
                label_time = (label_points['timestamp_ms'] - t0) / 1000.0
                for plot in [self.plots['biopac_ecg'], self.plots['biopac_eda'], self.plots['biopac_ppg']]:
                    plot.plot(label_time, np.zeros_like(label_time), 
                            pen=None, symbol='o', symbolBrush='r', 
                            symbolSize=10, name='Labels')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Set dark theme
    pg.setConfigOption('background', 'k')
    pg.setConfigOption('foreground', 'w')
    
    viewer = TimeSeriesViewer()
    viewer.show()
    sys.exit(app.exec_())