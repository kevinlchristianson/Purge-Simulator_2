import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QLineEdit, QComboBox,
    QHBoxLayout, QCheckBox, QRadioButton, QPushButton, QMessageBox, QStackedWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QFormLayout, QSpacerItem, QSizePolicy,
    QButtonGroup
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator


class PurgeSetupPage(QWidget):
    def __init__(self, profile_path=None):
        super().__init__()

        self.profile_path = profile_path
        self.total_length = 0.0
        self.total_points = 0

        main = QVBoxLayout()

        # Profile info
        top_row = QHBoxLayout()
        self.profile_path_label = QLabel(self.profile_path or "No appended profile loaded")
        top_row.addWidget(self.profile_path_label)
        top_row.addStretch(1)
        main.addLayout(top_row)

        self.profile_summary = QLabel("Final Profile Length: 0.00 miles | Points: 0")
        main.addWidget(self.profile_summary)

        # Compact form layout for inputs
        form = QFormLayout()

        self.start_mp_input = QLineEdit()
        self.start_mp_input.setValidator(QDoubleValidator())
        form.addRow("Purge Start MP:", self.start_mp_input)

        self.end_mp_input = QLineEdit()
        self.end_mp_input.setValidator(QDoubleValidator())
        form.addRow("Purge End MP:", self.end_mp_input)

        main.addLayout(form)

        # Intermediate pump stations checkbox (optional)
        self.has_pump_checkbox = QCheckBox("Intermediate Pump Stations")
        self.has_pump_checkbox.toggled.connect(self.toggle_pump_details)
        main.addWidget(self.has_pump_checkbox)

        self.pump_section = QWidget()
        pump_layout = QVBoxLayout()

        entry_row = QHBoxLayout()
        self.pump_mp_input = QLineEdit()
        self.pump_mp_input.setPlaceholderText("Milepost")
        self.pump_mp_input.setValidator(QDoubleValidator())
        self.pump_pressure_input = QLineEdit()
        self.pump_pressure_input.setPlaceholderText("Inlet Pressure (psi)")
        self.pump_pressure_input.setValidator(QDoubleValidator())
        self.add_pump_btn = QPushButton("Add")
        self.add_pump_btn.clicked.connect(self.add_pump)
        self.remove_pump_btn = QPushButton("Remove Selected")
        self.remove_pump_btn.clicked.connect(self.remove_selected_pump)

        entry_row.addWidget(self.pump_mp_input)
        entry_row.addWidget(self.pump_pressure_input)
        entry_row.addWidget(self.add_pump_btn)
        entry_row.addWidget(self.remove_pump_btn)

        pump_layout.addLayout(entry_row)

        self.pump_table = QTableWidget(0, 2)
        self.pump_table.setHorizontalHeaderLabels(["Milepost", "Inlet Pressure (psi)"])
        self.pump_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.pump_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.pump_table.setSelectionMode(QTableWidget.SingleSelection)
        pump_layout.addWidget(self.pump_table)

        self.pump_section.setLayout(pump_layout)
        self.pump_section.hide()
        main.addWidget(self.pump_section)

        # Endpoint MP selection (always required)
        endpoint_mp_row = QHBoxLayout()
        self.endpoint_mp_checkbox = QCheckBox("Profile End")
        self.endpoint_mp_input = QLineEdit()
        self.endpoint_mp_input.setValidator(QDoubleValidator())
        self.endpoint_mp_input.setPlaceholderText("Enter custom MP")
        self.endpoint_mp_checkbox.setChecked(True)
        self.endpoint_mp_input.setEnabled(False)
        self.endpoint_mp_checkbox.stateChanged.connect(self.toggle_endpoint_input)

        endpoint_mp_row.addWidget(QLabel("System Endpoint MP:"))
        endpoint_mp_row.addWidget(self.endpoint_mp_checkbox)
        endpoint_mp_row.addWidget(self.endpoint_mp_input)

        main.addLayout(endpoint_mp_row)

        # System endpoint dropdown row
        endpoint_row = QHBoxLayout()
        endpoint_row.addWidget(QLabel("System Endpoint:"))
        self.endpoint_combo = QComboBox()
        self.endpoint_combo.addItems([
            "Tankage",
            "Pump Station",
            "Merging with Downstream Pipeline",
        ])
        endpoint_row.addWidget(self.endpoint_combo)
        main.addLayout(endpoint_row)

        # Endpoint Pressure Behavior section (exclusive)
        behavior_row = QVBoxLayout()
        behavior_row.addWidget(QLabel("Endpoint Pressure Behavior:"))

        self.behavior_group = QButtonGroup(self)
        self.behavior_group.setExclusive(True)
        self.behavior_static = QRadioButton("Static Hold")
        self.behavior_dynamic = QRadioButton("Dynamic")
        self.behavior_throttle = QRadioButton("Throttle Down Near End")

        self.behavior_group.addButton(self.behavior_static)
        self.behavior_group.addButton(self.behavior_dynamic)
        self.behavior_group.addButton(self.behavior_throttle)

        self.behavior_static.toggled.connect(lambda checked: self.show_behavior_inputs("static") if checked else None)
        self.behavior_dynamic.toggled.connect(lambda checked: self.show_behavior_inputs("dynamic") if checked else None)
        self.behavior_throttle.toggled.connect(lambda checked: self.show_behavior_inputs("throttle") if checked else None)

        behavior_row.addWidget(self.behavior_static)
        behavior_row.addWidget(self.behavior_dynamic)
        behavior_row.addWidget(self.behavior_throttle)

        # Containers for dynamic inputs
        self.behavior_inputs = QWidget()
        self.behavior_inputs_layout = QVBoxLayout()
        self.behavior_inputs.setLayout(self.behavior_inputs_layout)
        self.behavior_inputs.setVisible(False)

        behavior_row.addWidget(self.behavior_inputs)

        self.behavior_info_btn = QPushButton("Behavior Info")
        self.behavior_info_btn.clicked.connect(self.show_behavior_info)
        behavior_row.addWidget(self.behavior_info_btn)

        main.addLayout(behavior_row)

        # Spacer
        main.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Navigation
        nav_layout = QHBoxLayout()
        self.back_btn = QPushButton("Back")
        self.next_btn = QPushButton("Next")
        nav_layout.addWidget(self.back_btn)
        nav_layout.addWidget(self.next_btn)
        main.addLayout(nav_layout)

        self.setLayout(main)

    def toggle_pump_details(self, checked):
        self.pump_section.setVisible(checked)

    def add_pump(self):
        mp = self.pump_mp_input.text().strip()
        pressure = self.pump_pressure_input.text().strip()
        if not mp or not pressure:
            QMessageBox.warning(self, "Missing Input", "Enter both Milepost and Inlet Pressure.")
            return
        row = self.pump_table.rowCount()
        self.pump_table.insertRow(row)
        self.pump_table.setItem(row, 0, QTableWidgetItem(mp))
        self.pump_table.setItem(row, 1, QTableWidgetItem(pressure))
        self.pump_mp_input.clear()
        self.pump_pressure_input.clear()

    def remove_selected_pump(self):
        row = self.pump_table.currentRow()
        if row >= 0:
            self.pump_table.removeRow(row)

    def toggle_endpoint_input(self, state):
        self.endpoint_mp_input.setEnabled(not state == Qt.Checked)

    def refresh_profile_summary(self, length=0.0, points=0):
        self.total_length = length
        self.total_points = points
        self.profile_summary.setText(
            f"Final Profile Length: {length:.2f} miles | Points: {points}"
        )

    def show_behavior_inputs(self, behavior_type):
        while self.behavior_inputs_layout.count():
            item = self.behavior_inputs_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if behavior_type == "static":
            self.behavior_inputs_layout.addWidget(QLabel("System Endpoint Pressure (psi):"))
            self.static_pressure_input = QLineEdit()
            self.static_pressure_input.setValidator(QDoubleValidator())
            self.behavior_inputs_layout.addWidget(self.static_pressure_input)

        elif behavior_type == "dynamic":
            self.behavior_inputs_layout.addWidget(QLabel("Maximum Endpoint Pressure (psi):"))
            self.dynamic_max_input = QLineEdit()
            self.dynamic_max_input.setValidator(QDoubleValidator())
            self.behavior_inputs_layout.addWidget(self.dynamic_max_input)
            self.behavior_inputs_layout.addWidget(QLabel("Minimum Endpoint Pressure (psi):"))
            self.dynamic_min_input = QLineEdit()
            self.dynamic_min_input.setValidator(QDoubleValidator())
            self.behavior_inputs_layout.addWidget(self.dynamic_min_input)

        elif behavior_type == "throttle":
            self.behavior_inputs_layout.addWidget(QLabel("Run Pressure (psi):"))
            self.throttle_run_input = QLineEdit()
            self.throttle_run_input.setValidator(QDoubleValidator())
            self.behavior_inputs_layout.addWidget(self.throttle_run_input)
            self.behavior_inputs_layout.addWidget(QLabel("Throttle Down Point (Miles from End):"))
            self.throttle_point_input = QLineEdit()
            self.throttle_point_input.setValidator(QDoubleValidator())
            self.behavior_inputs_layout.addWidget(self.throttle_point_input)
            self.behavior_inputs_layout.addWidget(QLabel("Throttled Pig Speed (mph):"))
            self.throttle_speed_input = QLineEdit()
            self.throttle_speed_input.setValidator(QDoubleValidator())
            self.behavior_inputs_layout.addWidget(self.throttle_speed_input)

        self.behavior_inputs.setVisible(True)

    def show_behavior_info(self):
        info = (
            "Static Hold: Maintains constant outlet pressure throughout purge. I.e. Purging into tankage; Pushing into adjacent pipeline.\n\n"
            "Dynamic: Pressure adjusts to control inlet flow, controlling outlet flow from exceeding maximum pig speed. I.e. Backpressure control valve; Downhill purge that would otherwise run away.\n\n"
            "Throttle Down Near End: During the majority of the purge, outlet pressure is maintained static at the 'Run' pressure to maximize pig speed. At a specified point (MP), pressure is switched to 'Dynamic' to control pig speed near the purge end. Ex: A purge all the way to tankage, where during the majority of the purge, pig speed is maximized. As the pig nears the end, it needs throttled down in velocity to keep nitrogen from entering the tank. Outlet pressure becomes dynamic, to hold back to the pig speed minimum."
        )
        QMessageBox.information(self, "Behavior Info", info)


class Wizard(QMainWindow):
    def __init__(self, profile_path=None, profile_length=0.0, profile_points=0):
        super().__init__()
        self.setWindowTitle("Purge Setup Wizard Packet")
        self.resize(700, 600)

        self.stack = QStackedWidget()

        self.purge_page = PurgeSetupPage(profile_path=profile_path)
        self.purge_page.refresh_profile_summary(profile_length, profile_points)

        self.stack.addWidget(self.purge_page)

        self.purge_page.back_btn.clicked.connect(self.on_back)
        self.purge_page.next_btn.clicked.connect(self.on_next)

        self.setCentralWidget(self.stack)

    def on_back(self):
        print("Back pressed (debug placeholder)")

    def on_next(self):
        data = {
            "purge_start": self.purge_page.start_mp_input.text(),
            "purge_end": self.purge_page.end_mp_input.text(),
            "has_pumps": self.purge_page.has_pump_checkbox.isChecked(),
            "pumps": [
                {
                    "mp": self.purge_page.pump_table.item(r, 0).text(),
                    "pressure": self.purge_page.pump_table.item(r, 1).text(),
                }
                for r in range(self.purge_page.pump_table.rowCount())
            ],
            "endpoint_mp": "profile_end" if self.purge_page.endpoint_mp_checkbox.isChecked() else self.purge_page.endpoint_mp_input.text(),
            "system_endpoint": self.purge_page.endpoint_combo.currentText(),
            "behavior": "static" if self.purge_page.behavior_static.isChecked() else "dynamic" if self.purge_page.behavior_dynamic.isChecked() else "throttle" if self.purge_page.behavior_throttle.isChecked() else None,
        }
        print("Collected Inputs:", data)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # For integration, values should be passed in from previous packets
    w = Wizard(profile_path=None, profile_length=0.0, profile_points=0)
    w.show()
    sys.exit(app.exec())
