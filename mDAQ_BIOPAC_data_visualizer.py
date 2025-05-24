"""
Physiological Data Viewer - An application for visualizing and comparing physiological data
from different data acquisition systems (mDAQ and BIOPAC).

Features:
- Load and view CSV files from mDAQ and BIOPAC systems
- Display multiple channels (ECG, EDA, PPG/IR) simultaneously
- Customizable visualization options (colors, labels, time windows)
- Label annotations with customizable appearance
- Synchronized time axis across all plots
"""

import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                           QGroupBox, QCheckBox, QSpinBox, QScrollArea,
                           QColorDialog, QComboBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
import pyqtgraph as pg

class TimeSeriesViewer(QMainWindow):
    """
    A PyQt5-based application for visualizing and comparing physiological data.
    
    This viewer allows for loading and display of time series data from mDAQ
    and BIOPAC systems, with support for multiple channels and customizable
    visualization options.
    """
    
    def __init__(self):
        """Initialize the TimeSeriesViewer application."""
        super().__init__()
        self.setWindowTitle('Physiological Data Viewer')
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize data storage
        self.mdaq_data = None
        self.biopac_data = None
        self.plots = {}
        self.plot_items = {}
        self.text_items = {}
        self.event_lines = {}
        
        # Initialize colors
        self.channel_colors = {
            'mdaq': {'ecg': '#00FF00', 'eda': '#FF0000', 'ir': '#0000FF'},
            'biopac': {'ECG': '#00FF00', 'EDA': '#FF0000', 'PPG': '#0000FF'}
        }
        self.background_color = 'k'
        self.grid_color = '#2C2C2C'
        self.text_color = 'w'  # Default text color for axes and titles
        self.label_colors = {
            'line': '#FF0000',
            'text': '#FF0000',
            'marker': '#FF0000'
        }
        
        # Initialize label visualization settings
        self.label_settings = {
            'show_lines': True,
            'show_text': True,
            'show_markers': True,
            'text_position': 'top',  # 'top', 'bottom', or 'middle'
            'line_style': Qt.DashLine,  # Qt.SolidLine, Qt.DashLine, Qt.DotLine
            'marker_size': 10,
            'marker_symbol': 't'  # 't', 'o', 's', 'd', '+'
        }
        
        # Setup a timer for delayed label updates to prevent performance issues
        # during continuous zooming or panning
        self.label_update_timer = QTimer()
        self.label_update_timer.setSingleShot(True)
        self.label_update_timer.timeout.connect(self.update_all_labels_delayed)
        self.current_view_changed = None
        
        # Setup UI
        self.setup_ui()

    def add_label_controls(self, layout):
        """
        Add controls for customizing label appearance.
        
        Args:
            layout: The parent layout to add controls to
        """
        label_group = QGroupBox("Label Visualization")
        label_layout = QVBoxLayout()
        
        # Checkboxes for label elements
        self.label_checkboxes = {
            'show_lines': QCheckBox("Show vertical lines"),
            'show_text': QCheckBox("Show text labels"),
            'show_markers': QCheckBox("Show markers")
        }
        for key, checkbox in self.label_checkboxes.items():
            checkbox.setChecked(self.label_settings[key])
            checkbox.stateChanged.connect(self.update_label_visibility)
            label_layout.addWidget(checkbox)
        
        # Text position
        text_pos_layout = QHBoxLayout()
        text_pos_layout.addWidget(QLabel("Label position:"))
        self.text_pos_combo = QComboBox()
        self.text_pos_combo.addItems(['top', 'middle', 'bottom'])
        self.text_pos_combo.setCurrentText(self.label_settings['text_position'])
        self.text_pos_combo.currentTextChanged.connect(self.update_label_position)
        text_pos_layout.addWidget(self.text_pos_combo)
        label_layout.addLayout(text_pos_layout)
        
        # Line style
        line_style_layout = QHBoxLayout()
        line_style_layout.addWidget(QLabel("Line style:"))
        self.line_style_combo = QComboBox()
        self.line_style_combo.addItems(['Solid', 'Dashed', 'Dotted'])
        
        # Set the current selection based on current style
        if self.label_settings['line_style'] == Qt.SolidLine:
            self.line_style_combo.setCurrentText('Solid')
        elif self.label_settings['line_style'] == Qt.DashLine:
            self.line_style_combo.setCurrentText('Dashed')
        else:
            self.line_style_combo.setCurrentText('Dotted')
            
        self.line_style_combo.currentTextChanged.connect(self.update_line_style)
        line_style_layout.addWidget(self.line_style_combo)
        label_layout.addLayout(line_style_layout)
        
        # Marker settings
        marker_layout = QHBoxLayout()
        marker_layout.addWidget(QLabel("Marker:"))
        self.marker_combo = QComboBox()
        self.marker_combo.addItems(['triangle', 'circle', 'square', 'diamond', 'plus'])
        
        # Set the current selection based on current symbol
        symbol_to_text = {'t': 'triangle', 'o': 'circle', 's': 'square', 'd': 'diamond', '+': 'plus'}
        current_symbol_text = symbol_to_text.get(self.label_settings['marker_symbol'], 'triangle')
        self.marker_combo.setCurrentText(current_symbol_text)
        
        self.marker_combo.currentTextChanged.connect(self.update_marker_style)
        marker_layout.addWidget(self.marker_combo)
        
        self.marker_size_spin = QSpinBox()
        self.marker_size_spin.setRange(5, 20)
        self.marker_size_spin.setValue(self.label_settings['marker_size'])
        self.marker_size_spin.valueChanged.connect(self.update_marker_size)
        marker_layout.addWidget(self.marker_size_spin)
        label_layout.addLayout(marker_layout)
        
        # Add "Reset Labels" button
        self.reset_labels_btn = QPushButton("Reset Label Positions")
        self.reset_labels_btn.clicked.connect(self.reset_label_positions)
        label_layout.addWidget(self.reset_labels_btn)
        
        label_group.setLayout(label_layout)
        layout.addWidget(label_group)
        
    def reset_label_positions(self):
        """
        Manually reset all label positions based on current viewport.
        This function forces a redraw of all labels using the current view settings.
        """
        # Force redraw of all labels for all plots
        if self.mdaq_data is not None and 'label' in self.mdaq_data['data'].columns:
            # Clear and replot mDAQ labels
            for key in self.plots:
                if key.startswith('mdaq'):
                    self.clear_label_items(key)
                    self.plot_labels_for_plot(self.mdaq_data['data'], self.mdaq_data['time'], 'mdaq', key)
                    
        if self.biopac_data is not None and 'label' in self.biopac_data['data'].columns:
            # Clear and replot BIOPAC labels
            for key in self.plots:
                if key.startswith('biopac'):
                    self.clear_label_items(key)
                    self.plot_labels_for_plot(self.biopac_data['data'], self.biopac_data['time'], 'biopac', key)

    def setup_ui(self):
        """Set up the main user interface components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create control panel
        control_panel = self.create_control_panel()
        scroll_area = QScrollArea()
        scroll_area.setWidget(control_panel)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(300)
        scroll_area.setMaximumWidth(400)
        main_layout.addWidget(scroll_area)
        
        # Create plot area
        plot_area = QWidget()
        plot_layout = QVBoxLayout(plot_area)
        self.plot_widget = pg.GraphicsLayoutWidget()
        plot_layout.addWidget(self.plot_widget)
        main_layout.addWidget(plot_area, stretch=4)
        
        # Setup initial plots
        self.setup_plots()
        
    def create_control_panel(self):
        """
        Create the control panel widget with all settings controls.
        
        Returns:
            QWidget: The control panel widget
        """
        control_panel = QWidget()
        layout = QVBoxLayout(control_panel)
        
        # File Loading Section
        file_group = QGroupBox("Load Data")
        file_layout = QVBoxLayout()
        
        # mDAQ controls
        self.mdaq_button = QPushButton('Load mDAQ File')
        self.mdaq_button.clicked.connect(lambda: self.load_file('mdaq'))
        self.mdaq_label = QLabel('No mDAQ file loaded')
        file_layout.addWidget(self.mdaq_button)
        file_layout.addWidget(self.mdaq_label)
        
        # BIOPAC controls
        self.biopac_button = QPushButton('Load BIOPAC File')
        self.biopac_button.clicked.connect(lambda: self.load_file('biopac'))
        self.biopac_label = QLabel('No BIOPAC file loaded')
        file_layout.addWidget(self.biopac_button)
        file_layout.addWidget(self.biopac_label)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Channel Selection
        channel_group = QGroupBox("Channel Selection")
        channel_layout = QVBoxLayout()
        
        # mDAQ channels
        self.mdaq_channels = {
            'ecg': QCheckBox('ECG'),
            'eda': QCheckBox('EDA'),
            'ir': QCheckBox('PPG (IR)')
        }
        mdaq_sub_group = QGroupBox("mDAQ Channels")
        mdaq_sub_layout = QVBoxLayout()
        for checkbox in self.mdaq_channels.values():
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.update_plot_visibility)
            mdaq_sub_layout.addWidget(checkbox)
        mdaq_sub_group.setLayout(mdaq_sub_layout)
        channel_layout.addWidget(mdaq_sub_group)
        
        # BIOPAC channels
        self.biopac_channels = {
            'ECG': QCheckBox('ECG'),
            'EDA': QCheckBox('EDA'),
            'PPG': QCheckBox('PPG')
        }
        biopac_sub_group = QGroupBox("BIOPAC Channels")
        biopac_sub_layout = QVBoxLayout()
        for checkbox in self.biopac_channels.values():
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.update_plot_visibility)
            biopac_sub_layout.addWidget(checkbox)
        biopac_sub_group.setLayout(biopac_sub_layout)
        channel_layout.addWidget(biopac_sub_group)
        
        channel_group.setLayout(channel_layout)
        layout.addWidget(channel_group)
        
        # Visualization Controls
        viz_group = QGroupBox("Visualization Controls")
        viz_layout = QVBoxLayout()
        
        # Time window control
        time_window_layout = QHBoxLayout()
        time_window_layout.addWidget(QLabel("Time Window (s):"))
        self.time_window_spin = QSpinBox()
        self.time_window_spin.setRange(1, 3600)
        self.time_window_spin.setValue(60)
        self.time_window_spin.valueChanged.connect(self.update_time_window)
        time_window_layout.addWidget(self.time_window_spin)
        viz_layout.addLayout(time_window_layout)
        
        # Auto-scale control
        self.auto_scale = QCheckBox("Auto Scale Y-Axis")
        self.auto_scale.setChecked(True)
        self.auto_scale.stateChanged.connect(self.toggle_auto_scale)
        viz_layout.addWidget(self.auto_scale)
        
        # Color controls
        self.add_color_controls(viz_layout)
        
        # Label visualization controls
        self.add_label_controls(viz_layout)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        return control_panel
    
    def create_color_picker(self, label_text, initial_color, callback):
        """
        Create a color picker layout with label and button.
        
        Args:
            label_text: Text for the label
            initial_color: Initial color for the button
            callback: Function to call when color is chosen
            
        Returns:
            QHBoxLayout: Layout containing the label and color button
        """
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))
        color_btn = QPushButton()
        color_btn.setFixedSize(50, 20)
        color_btn.setStyleSheet(f"background-color: {initial_color}")
        color_btn.clicked.connect(callback)
        layout.addWidget(color_btn)
        return layout

    def add_color_controls(self, layout):
        """
        Add color control widgets to the given layout.
        
        Args:
            layout: The parent layout to add controls to
        """
        color_group = QGroupBox("Color Settings")
        color_layout = QVBoxLayout()
        
        # Store color buttons for later reference
        self.color_buttons = {}
        
        # Background color
        bg_layout = QHBoxLayout()
        bg_layout.addWidget(QLabel("Background:"))
        self.bg_color_btn = QPushButton()
        self.bg_color_btn.setFixedSize(50, 20)
        self.bg_color_btn.setStyleSheet(f"background-color: {self.background_color}")
        self.bg_color_btn.clicked.connect(self.change_background_color)
        bg_layout.addWidget(self.bg_color_btn)
        color_layout.addLayout(bg_layout)
        
        # Grid color
        grid_layout = QHBoxLayout()
        grid_layout.addWidget(QLabel("Grid:"))
        self.grid_color_btn = QPushButton()
        self.grid_color_btn.setFixedSize(50, 20)
        self.grid_color_btn.setStyleSheet(f"background-color: {self.grid_color}")
        self.grid_color_btn.clicked.connect(self.change_grid_color)
        grid_layout.addWidget(self.grid_color_btn)
        color_layout.addLayout(grid_layout)
        
        # Text color - NEW
        text_layout = QHBoxLayout()
        text_layout.addWidget(QLabel("Text:"))
        self.text_color_btn = QPushButton()
        self.text_color_btn.setFixedSize(50, 20)
        self.text_color_btn.setStyleSheet(f"background-color: {self.text_color}")
        self.text_color_btn.clicked.connect(self.change_text_color)
        text_layout.addWidget(self.text_color_btn)
        color_layout.addLayout(text_layout)
        
        # Channel colors
        channel_group = QGroupBox("Channel Colors")
        channel_layout = QVBoxLayout()
        
        for device in ['mdaq', 'biopac']:
            for channel, color in self.channel_colors[device].items():
                btn_layout = QHBoxLayout()
                btn_layout.addWidget(QLabel(f"{device.upper()} {channel}:"))
                color_btn = QPushButton()
                color_btn.setFixedSize(50, 20)
                color_btn.setStyleSheet(f"background-color: {color}")
                
                # Store reference to button
                self.color_buttons[f"{device}_{channel}"] = color_btn
                
                # Use lambda with default arguments to prevent late binding issues
                color_btn.clicked.connect(
                    lambda checked, d=device, ch=channel: self.change_channel_color(d, ch)
                )
                btn_layout.addWidget(color_btn)
                channel_layout.addLayout(btn_layout)
        
        channel_group.setLayout(channel_layout)
        color_layout.addWidget(channel_group)
        
        # Label colors
        label_group = QGroupBox("Label Colors")
        label_layout = QVBoxLayout()
        
        for item, color in self.label_colors.items():
            btn_layout = QHBoxLayout()
            btn_layout.addWidget(QLabel(f"Label {item}:"))
            color_btn = QPushButton()
            color_btn.setFixedSize(50, 20)
            color_btn.setStyleSheet(f"background-color: {color}")
            
            # Store reference to button
            self.color_buttons[f"label_{item}"] = color_btn
            
            # Use lambda with default arguments
            color_btn.clicked.connect(
                lambda checked, i=item: self.change_label_color(i)
            )
            btn_layout.addWidget(color_btn)
            label_layout.addLayout(btn_layout)
        
        label_group.setLayout(label_layout)
        color_layout.addWidget(label_group)
        
        color_group.setLayout(color_layout)
        layout.addWidget(color_group)

    def setup_plots(self):
        """
        Initialize and configure all plots for the application.
        Sets up plot structure, legend, grid, and links X axes.
        """
        self.plots = {}
        self.plot_items = {}
        self.text_items = {}
        self.event_lines = {}
        
        # Create plots for each channel
        for i, channel in enumerate(['ecg', 'eda', 'ir']):
            # mDAQ plot
            self.plots[f'mdaq_{channel}'] = self.plot_widget.addPlot(row=i, col=0, 
                title=f'mDAQ {channel.upper()}')
            self.text_items[f'mdaq_{channel}'] = []
            self.event_lines[f'mdaq_{channel}'] = []
            
            # BIOPAC plot
            biopac_channel = channel.upper() if channel != 'ir' else 'PPG'
            self.plots[f'biopac_{biopac_channel}'] = self.plot_widget.addPlot(row=i, col=1,
                title=f'BIOPAC {biopac_channel}')
            self.text_items[f'biopac_{biopac_channel}'] = []
            self.event_lines[f'biopac_{biopac_channel}'] = []
            
            # Configure plots
            for plot in [self.plots[f'mdaq_{channel}'], 
                        self.plots[f'biopac_{biopac_channel}']]:
                plot.showGrid(x=True, y=True)
                plot.addLegend()
                
                # Set text colors
                plot.getAxis('bottom').setTextPen(self.text_color)
                plot.getAxis('left').setTextPen(self.text_color)
                plot.setTitle(plot.titleLabel.text, color=self.text_color)
                
                # Connect view changed signal to update label positions
                plot.sigRangeChanged.connect(self.on_view_changed)
        
        # Link all x-axes to the first plot
        first_plot = list(self.plots.values())[0]
        for plot in self.plots.values():
            if plot != first_plot:
                plot.setXLink(first_plot)
    
    def on_view_changed(self, view):
        """
        Handler for when the plot view changes (zoom, pan, etc.)
        Schedules an update of label positions after a short delay to prevent
        performance issues during continuous zooming/panning.
        
        Args:
            view: The view widget that changed
        """
        # Store the view that changed
        self.current_view_changed = view
        
        # Restart the timer - this prevents excessive updates during rapid interactions
        self.label_update_timer.start(100)  # 100ms delay
        
    def update_all_labels_delayed(self):
        """
        Updates all label positions after view changes, with debouncing to
        prevent performance issues during continuous zooming/panning.
        """
        if self.current_view_changed is None:
            return
            
        view = self.current_view_changed
        
        # Get the plot that changed
        for key, plot in self.plots.items():
            if plot == view:
                # Check if this plot has label data stored
                if hasattr(plot, 'plot_label_data'):
                    # Retrieve the stored data
                    plot_data = plot.plot_label_data
                    
                    # Update the label positions based on the new view
                    if 'device_type' in plot_data:
                        # Clear existing labels
                        self.clear_label_items(key)
                        
                        # Get current plot data from the appropriate source
                        if plot_data['device_type'] == 'mdaq' and self.mdaq_data is not None:
                            data = self.mdaq_data['data']
                            time = self.mdaq_data['time']
                        elif plot_data['device_type'] == 'biopac' and self.biopac_data is not None:
                            data = self.biopac_data['data']
                            time = self.biopac_data['time']
                        else:
                            continue
                        
                        # Re-draw the labels with updated positions
                        if 'label' in data.columns:
                            self.plot_labels_for_plot(data, time, plot_data['device_type'], key)
                            
        # Reset the current view changed
        self.current_view_changed = None
                    
    def load_file(self, device_type):
        """
        Load a CSV file for the specified device type.
        
        Args:
            device_type: Type of device ('mdaq' or 'biopac')
        """
        filename, _ = QFileDialog.getOpenFileName(self, f'Select {device_type.upper()} File',
            '', 'CSV Files (*.csv)')
        
        if filename:
            try:
                df = pd.read_csv(filename)
                
                # Convert timestamp to relative time
                t0 = df['timestamp_ms'].iloc[0]
                time = (df['timestamp_ms'] - t0) / 1000.0
                
                if device_type == 'mdaq':
                    self.mdaq_data = {'time': time, 'data': df}
                    self.mdaq_label.setText(f'mDAQ file: {filename.split("/")[-1]}')
                    self.clear_plot_items('mdaq')
                    self.plot_mdaq_data()
                else:
                    self.biopac_data = {'time': time, 'data': df}
                    self.biopac_label.setText(f'BIOPAC file: {filename.split("/")[-1]}')
                    self.clear_plot_items('biopac')
                    self.plot_biopac_data()
                
                # Plot labels if they exist
                if 'label' in df.columns:
                    self.plot_labels(df, time, device_type)
                
            except Exception as e:
                print(f"Error loading file: {e}")
    
    def clear_plot_items(self, device_type):
        """
        Clear all plotted items for the specified device type.
        
        Args:
            device_type: Type of device ('mdaq' or 'biopac')
        """
        # Clear plots for the specified device
        for key, plot in self.plots.items():
            if key.startswith(device_type):
                plot.clear()
                plot.addLegend()
                
                # Clear stored text items and event lines
                for text in self.text_items[key]:
                    plot.removeItem(text)
                self.text_items[key] = []
                
                for line in self.event_lines[key]:
                    plot.removeItem(line)
                self.event_lines[key] = []
    
    def plot_mdaq_data(self):
        """Plot mDAQ data if available."""
        if self.mdaq_data is None:
            return
            
        time = self.mdaq_data['time']
        df = self.mdaq_data['data']
        
        channels = {'ecg': 'ecg', 'eda': 'eda', 'ir': 'ir'}
        for channel, col in channels.items():
            if col in df.columns:
                plot = self.plots[f'mdaq_{channel}']
                color = self.channel_colors['mdaq'][channel]
                self.plot_items[f'mdaq_{channel}'] = plot.plot(
                    time, df[col], pen=color, name=col.upper())
    
    def plot_biopac_data(self):
        """Plot BIOPAC data if available."""
        if self.biopac_data is None:
            return
            
        time = self.biopac_data['time']
        df = self.biopac_data['data']
        
        channels = {'ECG': 'ECG', 'EDA': 'EDA', 'PPG': 'PPG'}
        for channel, col in channels.items():
            if col in df.columns:
                plot = self.plots[f'biopac_{channel}']
                color = self.channel_colors['biopac'][channel]
                self.plot_items[f'biopac_{channel}'] = plot.plot(
                    time, df[col], pen=color, name=channel)
    
    def plot_labels(self, df, time, device_type):
        """
        Plot labels for all plots of the specified device type.
        
        Args:
            df: DataFrame containing the data
            time: Time array corresponding to the data
            device_type: Type of device ('mdaq' or 'biopac')
        """
        # Plot labels on each applicable plot
        for key, plot in self.plots.items():
            if key.startswith(device_type):
                self.plot_labels_for_plot(df, time, device_type, key)
    
    def plot_labels_for_plot(self, df, time, device_type, plot_key):
        """
        Plot labels for a specific plot.
        
        Args:
            df: DataFrame containing the data
            time: Time array corresponding to the data
            device_type: Type of device ('mdaq' or 'biopac')
            plot_key: Key of the specific plot
        """
        # Get label points
        label_points = df[df['label'].notna()]
        if not label_points.empty:
            label_time = time[label_points.index]
            label_values = label_points['label']
            
            plot = self.plots[plot_key]
            # Clear existing items
            self.clear_label_items(plot_key)
            
            # Get current plot range for positioning
            view_range = plot.viewRange()
            y_min, y_max = view_range[1]
            y_range = y_max - y_min
            
            # Store data for this plot to enable update on viewport changes
            plot_data = {
                'label_time': label_time,
                'label_values': label_values,
                'device_type': device_type
            }
            # Store this data as an attribute of the plot for later use
            plot.plot_label_data = plot_data
            
            # Add vertical lines
            if self.label_settings['show_lines']:
                for t in label_time:
                    line = pg.InfiniteLine(
                        pos=t, 
                        angle=90, 
                        pen=pg.mkPen(color=self.label_colors['line'], 
                                   width=1, 
                                   style=self.label_settings['line_style'])
                    )
                    plot.addItem(line)
                    self.event_lines[plot_key].append(line)
            
            # Add text labels
            if self.label_settings['show_text']:
                for t, label in zip(label_time, label_values):
                    # Process label - remove everything after '@' character
                    processed_label = str(label)
                    if '@' in processed_label:
                        processed_label = processed_label.split('@')[0]
                    
                    # Position text based on settings
                    if self.label_settings['text_position'] == 'top':
                        y_pos = y_max - (0.1 * y_range)
                    elif self.label_settings['text_position'] == 'bottom':
                        y_pos = y_min + (0.1 * y_range)
                    else:  # middle
                        y_pos = y_min + (0.5 * y_range)
                    
                    text = pg.TextItem(
                        text=processed_label,
                        color=self.label_colors['text'],
                        anchor=(0.5, 0.5)
                    )
                    plot.addItem(text)
                    text.setPos(t, y_pos)
                    self.text_items[plot_key].append(text)
            
            # Add markers
            if self.label_settings['show_markers']:
                marker_pos = y_min + (0.05 * y_range)
                scatter = plot.plot(
                    label_time,
                    np.zeros_like(label_time) + marker_pos,
                    pen=None,
                    symbol=self.label_settings['marker_symbol'],
                    symbolBrush=self.label_colors['marker'],
                    symbolPen=None,
                    symbolSize=self.label_settings['marker_size']
                )
                # Store reference to scatter plot to remove later
                self.event_lines[plot_key].append(scatter)

    def clear_label_items(self, key):
        """
        Clear all label-related items for a specific plot.
        
        Args:
            key: Key of the plot to clear items for
        """
        plot = self.plots[key]
        
        # Clear text items
        for text in self.text_items[key]:
            plot.removeItem(text)
        self.text_items[key] = []
        
        # Clear event lines and markers
        for item in self.event_lines[key]:
            plot.removeItem(item)
        self.event_lines[key] = []

    def update_label_visibility(self):
        """Update label visibility based on checkbox settings."""
        for key, value in self.label_checkboxes.items():
            self.label_settings[key] = value.isChecked()
        self.replot_all_labels()

    def update_label_position(self, position):
        """
        Update the position of all label text.
        
        Args:
            position: New position setting ('top', 'middle', or 'bottom')
        """
        self.label_settings['text_position'] = position
        self.replot_all_labels()

    def update_line_style(self, style):
        """
        Update the style of label lines.
        
        Args:
            style: New line style ('Solid', 'Dashed', or 'Dotted')
        """
        style_map = {
            'Solid': Qt.SolidLine,
            'Dashed': Qt.DashLine,
            'Dotted': Qt.DotLine
        }
        self.label_settings['line_style'] = style_map[style]
        self.replot_all_labels()

    def update_marker_style(self, style):
        """
        Update the style of label markers.
        
        Args:
            style: New marker style
        """
        style_map = {
            'triangle': 't',
            'circle': 'o',
            'square': 's',
            'diamond': 'd',
            'plus': '+'
        }
        self.label_settings['marker_symbol'] = style_map[style]
        self.replot_all_labels()

    def update_marker_size(self, size):
        """
        Update the size of label markers.
        
        Args:
            size: New marker size
        """
        self.label_settings['marker_size'] = size
        self.replot_all_labels()

    def replot_all_labels(self):
        """Replot all labels with current settings."""
        if self.mdaq_data is not None and 'label' in self.mdaq_data['data'].columns:
            self.plot_labels(self.mdaq_data['data'], self.mdaq_data['time'], 'mdaq')
        if self.biopac_data is not None and 'label' in self.biopac_data['data'].columns:
            self.plot_labels(self.biopac_data['data'], self.biopac_data['time'], 'biopac')
    
    def change_grid_color(self):
        """
        Change the grid color for all plots using a color dialog.
        Updates the stored color and applies it to all plots.
        """
        color = QColorDialog.getColor()
        if color.isValid():
            self.grid_color = color.name()
            self.grid_color_btn.setStyleSheet(f"background-color: {self.grid_color}")
            
            for plot in self.plots.values():
                # Update grid lines
                plot.getAxis('bottom').setPen(self.grid_color)
                plot.getAxis('left').setPen(self.grid_color)
                plot.showGrid(x=True, y=True, alpha=0.3)
    
    def change_text_color(self):
        """
        Change the text color for all plots using a color dialog.
        Updates text color for axes labels, titles, and legend.
        """
        color = QColorDialog.getColor()
        if color.isValid():
            self.text_color = color.name()
            self.text_color_btn.setStyleSheet(f"background-color: {self.text_color}")
            
            for plot in self.plots.values():
                # Update text color for axes
                plot.getAxis('bottom').setTextPen(self.text_color)
                plot.getAxis('left').setTextPen(self.text_color)
                
                # Update title color
                title_text = plot.titleLabel.text
                plot.setTitle(title_text, color=self.text_color)
                
                # Update legend text color if possible
                if hasattr(plot, 'legend') and plot.legend is not None:
                    for item in plot.legend.items:
                        if hasattr(item[1], 'setText'):
                            item[1].setText(item[1].text, color=self.text_color)
    
    def change_label_color(self, item):
        """
        Change color for label elements (lines, text, or markers).
        
        Args:
            item: Type of label element to change color for
        """
        color = QColorDialog.getColor()
        if color.isValid():
            self.label_colors[item] = color.name()
            # Update color button
            btn_key = f"label_{item}"
            if btn_key in self.color_buttons:
                self.color_buttons[btn_key].setStyleSheet(f"background-color: {color.name()}")
            self.replot_all_labels()
    
    def update_plot_visibility(self):
        """
        Update the visibility of plots based on channel checkbox states.
        """
        # Update mDAQ plot visibility
        for channel, checkbox in self.mdaq_channels.items():
            if f'mdaq_{channel}' in self.plots:
                self.plots[f'mdaq_{channel}'].setVisible(checkbox.isChecked())
        
        # Update BIOPAC plot visibility
        for channel, checkbox in self.biopac_channels.items():
            if f'biopac_{channel}' in self.plots:
                self.plots[f'biopac_{channel}'].setVisible(checkbox.isChecked())
    
    def toggle_auto_scale(self, state):
        """
        Toggle automatic scaling of the Y-axis.
        
        Args:
            state: Checkbox state (True for auto-scale, False for fixed scale)
        """
        for plot in self.plots.values():
            plot.enableAutoRange('y' if state else None)
    
    def update_time_window(self, value):
        """
        Update the visible time window for all plots.
        
        Args:
            value: Time window in seconds
        """
        if len(self.plots) > 0:
            first_plot = list(self.plots.values())[0]
            first_plot.setXRange(-value, 0, padding=0)
    
    def change_background_color(self):
        """
        Change the background color for all plots using a color dialog.
        Updates the stored color and applies it to the plot widget.
        """
        color = QColorDialog.getColor()
        if color.isValid():
            self.background_color = color.name()
            self.bg_color_btn.setStyleSheet(f"background-color: {self.background_color}")
            self.plot_widget.setBackground(self.background_color)
    
    def change_channel_color(self, device, channel):
        """
        Change color for a specific channel in a specific device.
        
        Args:
            device: Device type ('mdaq' or 'biopac')
            channel: Channel name
        """
        color = QColorDialog.getColor()
        if color.isValid():
            self.channel_colors[device][channel] = color.name()
            # Update color button
            btn_key = f"{device}_{channel}"
            if btn_key in self.color_buttons:
                self.color_buttons[btn_key].setStyleSheet(f"background-color: {color.name()}")
            # Replot data with new color
            if device == 'mdaq' and self.mdaq_data is not None:
                self.plot_mdaq_data()
            elif device == 'biopac' and self.biopac_data is not None:
                self.plot_biopac_data()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Set dark theme
    pg.setConfigOption('background', 'k')
    pg.setConfigOption('foreground', 'w')
    
    viewer = TimeSeriesViewer()
    viewer.show()
    sys.exit(app.exec_())