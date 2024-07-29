import sys
import os
import subprocess
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QComboBox, QTabWidget,
                             QTextEdit, QProgressBar, QLineEdit, QCheckBox, QSpinBox)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal

class WorkerThread(QThread):
    update_output = pyqtSignal(str)

    def __init__(self, command_args):
        super().__init__()
        self.command_args = command_args

    def run(self):
        process = subprocess.Popen(self.command_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                self.update_output.emit(output.strip())

        rc = process.poll()
        if rc != 0:
            error = process.stderr.read()
            self.update_output.emit(f"Error: {error}")

class FaceDetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SnapScan")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        self.setup_detection_tab()
        self.setup_matching_tab()

        self.setStyleSheet("""
            QWidget {
                background-color: #2D2D2D;
                color: #FFFFFF;
            }
            QPushButton {
                background-color: #4A4A4A;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #5A5A5A;
            }
            QComboBox, QSpinBox {
                background-color: #4A4A4A;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
            QTextEdit, QLineEdit {
                background-color: #3D3D3D;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #5A5A5A;
                background-color: #2D2D2D;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #5A5A5A;
                background-color: #4A90E2;
            }
            QTabWidget::pane {
                border: 1px solid #5A5A5A;
            }
            QTabBar::tab {
                background-color: #3D3D3D;
                padding: 5px 10px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #4A4A4A;
            }
        """)

    def setup_detection_tab(self):
        detection_tab = QWidget()
        layout = QVBoxLayout(detection_tab)

        input_layout = QHBoxLayout()
        self.input_path = QLineEdit()
        input_button = QPushButton("Select Input")
        input_button.clicked.connect(self.select_input)
        input_layout.addWidget(QLabel("Input:"))
        input_layout.addWidget(self.input_path)
        input_layout.addWidget(input_button)
        layout.addLayout(input_layout)

        output_layout = QHBoxLayout()
        self.output_path = QLineEdit()
        self.output_path.setText("output")  # Set default output folder
        output_button = QPushButton("Select Output")
        output_button.clicked.connect(self.select_output)
        output_layout.addWidget(QLabel("Output:"))
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(output_button)
        layout.addLayout(output_layout)

        mode_layout = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["OpenCV", "YuNet", "RetinaFace"])
        mode_layout.addWidget(QLabel("Mode:"))
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)

        timeframe_layout = QHBoxLayout()
        self.timeframe_check = QCheckBox("Use Timeframe")
        self.timeframe_start = QLineEdit()
        self.timeframe_end = QLineEdit()
        timeframe_layout.addWidget(self.timeframe_check)
        timeframe_layout.addWidget(QLabel("Start (MM.SS):"))
        timeframe_layout.addWidget(self.timeframe_start)
        timeframe_layout.addWidget(QLabel("End (MM.SS):"))
        timeframe_layout.addWidget(self.timeframe_end)
        layout.addLayout(timeframe_layout)

        faces_layout = QHBoxLayout()
        self.faces_check = QCheckBox("Limit Number of Faces")
        self.faces_number = QSpinBox()
        self.faces_number.setRange(1, 1000)
        self.faces_number.setValue(10)  # Set a default value
        faces_layout.addWidget(self.faces_check)
        faces_layout.addWidget(self.faces_number)
        layout.addLayout(faces_layout)

        self.no_folder_check = QCheckBox("No Folder (Save all outputs in the main output directory)")
        layout.addWidget(self.no_folder_check)

        self.detect_button = QPushButton("Detect Faces")
        self.detect_button.clicked.connect(self.detect_faces)
        layout.addWidget(self.detect_button)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        layout.addWidget(self.output_text)

        self.tabs.addTab(detection_tab, "Face Detection")

    def setup_matching_tab(self):
        matching_tab = QWidget()
        layout = QVBoxLayout(matching_tab)

        input_layout = QHBoxLayout()
        self.match_input_path = QLineEdit()
        input_button = QPushButton("Select Image")
        input_button.clicked.connect(self.select_match_input)
        input_layout.addWidget(QLabel("Input Image:"))
        input_layout.addWidget(self.match_input_path)
        input_layout.addWidget(input_button)
        layout.addLayout(input_layout)

        mode_layout = QHBoxLayout()
        self.match_mode_combo = QComboBox()
        self.match_mode_combo.addItems(["OpenCV", "YuNet", "RetinaFace"])
        mode_layout.addWidget(QLabel("Mode:"))
        mode_layout.addWidget(self.match_mode_combo)
        layout.addLayout(mode_layout)

        self.match_button = QPushButton("Match Face")
        self.match_button.clicked.connect(self.match_face)
        layout.addWidget(self.match_button)

        self.match_output_text = QTextEdit()
        self.match_output_text.setReadOnly(True)
        self.match_output_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        layout.addWidget(self.match_output_text)

        self.tabs.addTab(matching_tab, "Face Matching")

    def select_input(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Input File")
        if file_name:
            self.input_path.setText(file_name)

    def select_output(self):
        folder_name = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder_name:
            self.output_path.setText(folder_name)

    def select_match_input(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Input Image")
        if file_name:
            self.match_input_path.setText(file_name)

    def detect_faces(self):
        input_path = self.input_path.text()
        output_path = self.output_path.text()
        mode = self.mode_combo.currentIndex() + 1

        if not input_path:
            self.output_text.setText("Please select an input file.")
            return

        command_args = ["python", "facedetector.py", "-i", input_path, "-o", output_path, "-mode", str(mode)]

        if self.timeframe_check.isChecked():
            start_time = self.timeframe_start.text()
            end_time = self.timeframe_end.text()
            if start_time and end_time:
                command_args.extend(["-r", f"{start_time}-{end_time}"])

        if self.faces_check.isChecked():
            num_faces = self.faces_number.value()
            command_args.extend(["-n", str(num_faces)])

        if self.no_folder_check.isChecked():
            command_args.append("-nf")

        self.output_text.clear()
        self.worker_thread = WorkerThread(command_args)
        self.worker_thread.update_output.connect(self.update_output)
        self.worker_thread.start()

    def match_face(self):
        input_path = self.match_input_path.text()
        mode = self.match_mode_combo.currentIndex() + 1

        if not input_path:
            self.match_output_text.setText("Please select an input image.")
            return

        command_args = ["python", "facedetector.py", "-i", input_path, "-m", "-mode", str(mode)]
        self.match_output_text.clear()
        self.worker_thread = WorkerThread(command_args)
        self.worker_thread.update_output.connect(self.update_match_output)
        self.worker_thread.start()

    def update_output(self, text):
        self.output_text.append(text)

    def update_match_output(self, text):
        self.match_output_text.append(text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceDetectorGUI()
    window.show()
    sys.exit(app.exec())