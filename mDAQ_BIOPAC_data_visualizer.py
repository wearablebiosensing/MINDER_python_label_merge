import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                           QGroupBox, QCheckBox, QSpinBox, QScrollArea,
                           QColorDialog, QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import pyqtgraph as pg

class TimeSeriesViewer(QMainWindow):
    def __init__(self):
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
        self.text_color = 'w'
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
        
        # Setup UI
        self.setup_ui()

    def add_label_controls(self, layout):
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
        self.line_style_combo.currentTextChanged.connect(self.update_line_style)
        line_style_layout.addWidget(self.line_style_combo)
        label_layout.addLayout(line_style_layout)
        
        # Marker settings
        marker_layout = QHBoxLayout()
        marker_layout.addWidget(QLabel("Marker:"))
        self.marker_combo = QComboBox()
        self.marker_combo.addItems(['triangle', 'circle', 'square', 'diamond', 'plus'])
        self.marker_combo.currentTextChanged.connect(self.update_marker_style)
        marker_layout.addWidget(self.marker_combo)
        
        self.marker_size_spin = QSpinBox()
        self.marker_size_spin.setRange(5, 20)
        self.marker_size_spin.setValue(self.label_settings['marker_size'])
        self.marker_size_spin.valueChanged.connect(self.update_marker_size)
        marker_layout.addWidget(self.marker_size_spin)
        label_layout.addLayout(marker_layout)
        
        label_group.setLayout(label_layout)
        layout.addWidget(label_group)

        
    def setup_ui(self):
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
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))
        color_btn = QPushButton()
        color_btn.setFixedSize(50, 20)
        color_btn.setStyleSheet(f"background-color: {initial_color}")
        color_btn.clicked.connect(callback)
        layout.addWidget(color_btn)
        return layout



    
    def add_color_controls(self, layout):
        """Add color control widgets with improved initialization"""
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
        
        # Link all x-axes to the first plot
        first_plot = list(self.plots.values())[0]
        for plot in self.plots.values():
            if plot != first_plot:
                plot.setXLink(first_plot)
    
    def load_file(self, device_type):
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
        """Plot labels with improved visibility"""
        label_points = df[df['label'].notna()]
        if not label_points.empty:
            label_time = time[label_points.index]
            label_values = label_points['label']
            
            # Add label markers and text to relevant plots
            for key, plot in self.plots.items():
                if key.startswith(device_type):
                    # Clear existing items
                    self.clear_label_items(key)
                    
                    # Get plot range for positioning
                    view_range = plot.viewRange()
                    y_min, y_max = view_range[1]
                    y_range = y_max - y_min
                    
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
                            self.event_lines[key].append(line)
                    
                    # Add text labels
                    if self.label_settings['show_text']:
                        for t, label in zip(label_time, label_values):
                            # Position text based on settings
                            if self.label_settings['text_position'] == 'top':
                                y_pos = y_max - (0.1 * y_range)
                            elif self.label_settings['text_position'] == 'bottom':
                                y_pos = y_min + (0.1 * y_range)
                            else:  # middle
                                y_pos = y_min + (0.5 * y_range)
                            
                            text = pg.TextItem(
                                text=str(label),
                                color=self.label_colors['text'],
                                anchor=(0.5, 0.5)
                            )
                            plot.addItem(text)
                            text.setPos(t, y_pos)
                            self.text_items[key].append(text)
                    
                    # Add markers
                    if self.label_settings['show_markers']:
                        marker_pos = y_min + (0.05 * y_range)
                        plot.plot(
                            label_time,
                            np.zeros_like(label_time) + marker_pos,
                            pen=None,
                            symbol=self.label_settings['marker_symbol'],
                            symbolBrush=self.label_colors['marker'],
                            symbolPen=None,
                            symbolSize=self.label_settings['marker_size']
                        )

    def clear_label_items(self, key):
        """Clear all label-related items for a specific plot"""
        plot = self.plots[key]
        
        # Clear text items
        for text in self.text_items[key]:
            plot.removeItem(text)
        self.text_items[key] = []
        
        # Clear event lines
        for line in self.event_lines[key]:
            plot.removeItem(line)
        self.event_lines[key] = []

    def position_label_text(self, text_item, x_pos, plot):
        view_range = plot.viewRange()
        y_min, y_max = view_range[1]
        y_range = y_max - y_min
        
        if self.label_settings['text_position'] == 'top':
            y_pos = y_max - (0.05 * y_range)
        elif self.label_settings['text_position'] == 'bottom':
            y_pos = y_min + (0.05 * y_range)
        else:  # middle
            y_pos = y_min + (0.5 * y_range)
            
        text_item.setPos(x_pos, y_pos)

    def update_label_visibility(self):
        for key, value in self.label_checkboxes.items():
            self.label_settings[key] = value.isChecked()
        self.replot_all_labels()

    def update_label_position(self, position):
        self.label_settings['text_position'] = position
        self.replot_all_labels()

    def update_line_style(self, style):
        style_map = {
            'Solid': Qt.SolidLine,
            'Dashed': Qt.DashLine,
            'Dotted': Qt.DotLine
        }
        self.label_settings['line_style'] = style_map[style]
        self.replot_all_labels()

    def update_marker_style(self, style):
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
        self.label_settings['marker_size'] = size
        self.replot_all_labels()

    def replot_all_labels(self):
        if self.mdaq_data is not None:
            self.plot_labels(self.mdaq_data['data'], self.mdaq_data['time'], 'mdaq')
        if self.biopac_data is not None:
            self.plot_labels(self.biopac_data['data'], self.biopac_data['time'], 'biopac')

    def change_grid_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.grid_color = color.name()
            for plot in self.plots.values():
                plot.getAxis('bottom').setPen(self.grid_color)
                plot.getAxis('left').setPen(self.grid_color)
                plot.showGrid(x=True, y=True, alpha=0.3)

    def change_text_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.text_color = color.name()
            for plot in self.plots.values():
                plot.getAxis('bottom').setTextPen(self.text_color)
                plot.getAxis('left').setTextPen(self.text_color)
                plot.setTitle(plot.titleLabel.text, color=self.text_color)

    def change_label_color(self, item):
        """Update label color with button update"""
        color = QColorDialog.getColor()
        if color.isValid():
            self.label_colors[item] = color.name()
            # Update color button
            btn_key = f"label_{item}"
            if btn_key in self.color_buttons:
                self.color_buttons[btn_key].setStyleSheet(f"background-color: {color.name()}")
            self.replot_all_labels()



    
    def update_plot_visibility(self):
        # Update mDAQ plot visibility
        for channel, checkbox in self.mdaq_channels.items():
            if f'mdaq_{channel}' in self.plots:
                self.plots[f'mdaq_{channel}'].setVisible(checkbox.isChecked())
        
        # Update BIOPAC plot visibility
        for channel, checkbox in self.biopac_channels.items():
            if f'biopac_{channel}' in self.plots:
                self.plots[f'biopac_{channel}'].setVisible(checkbox.isChecked())
    
    def toggle_auto_scale(self, state):
        for plot in self.plots.values():
            plot.enableAutoRange('y' if state else None)
    
    def update_time_window(self, value):
        if len(self.plots) > 0:
            first_plot = list(self.plots.values())[0]
            first_plot.setXRange(-value, 0, padding=0)
    
    def change_background_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.background_color = color.name()
            self.bg_color_btn.setStyleSheet(f"background-color: {self.background_color}")
            self.plot_widget.setBackground(self.background_color)
    
    def change_channel_color(self, device, channel):
        """Update channel color with button update"""
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
