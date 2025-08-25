from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit,
    QComboBox, QPushButton, QFormLayout
)


class PipePage(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()

        title = QLabel("Pipe & Fluid Properties")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        form_layout = QFormLayout()

        # Pipe inputs
        self.nps_input = QLineEdit()
        self.nps_input.setPlaceholderText("Enter Nominal Pipe Size (in)")
        form_layout.addRow("NPS (in):", self.nps_input)

        self.wt_input = QLineEdit()
        self.wt_input.setPlaceholderText("Enter Wall Thickness (in)")
        self.wt_input.setText("0.250")  # Default
        form_layout.addRow("Wall Thickness (in):", self.wt_input)

        # Roughness options
        self.roughness_combo = QComboBox()
        self.roughness_options = {
            "Rusted/Corroded Welded Steel (0.0005 ft)": 0.0005,
            "New Welded Steel (0.00015 ft)": 0.00015,
            "Welded HDPE (0.000005 ft)": 0.000005,
        }
        for label in self.roughness_options:
            self.roughness_combo.addItem(label)
        form_layout.addRow("Pipe Roughness:", self.roughness_combo)

        # Fluid options
        self.fluid_combo = QComboBox()
        self.fluid_combo.addItems(["Diesel", "Gasoline", "Crude Oil", "Water", "NGL"])
        self.fluid_combo.setCurrentText("Diesel")  # Default fluid
        self.fluid_combo.currentTextChanged.connect(self.toggle_api_input)
        form_layout.addRow("Fluid:", self.fluid_combo)

        # API Gravity input (hidden unless Crude Oil selected)
        self.api_label = QLabel("API Gravity (only if Crude Oil):")
        self.api_input = QLineEdit()
        self.api_input.setPlaceholderText("Enter API Gravity")
        self.api_label.hide()
        self.api_input.hide()
        form_layout.addRow(self.api_label, self.api_input)

        layout.addLayout(form_layout)

        # Continue button (wizard will connect this later)
        self.continue_button = QPushButton("Next →")
        layout.addWidget(self.continue_button)

        self.setLayout(layout)

    def toggle_api_input(self, fluid):
        if fluid == "Crude Oil":
            self.api_label.show()
            self.api_input.show()
        else:
            self.api_label.hide()
            self.api_input.hide()

    def get_data(self):
        data = {
            "nps": self.nps_input.text(),
            "wt": self.wt_input.text(),
            "roughness": self.roughness_combo.currentText(),
            "roughness_value": self.roughness_options[self.roughness_combo.currentText()],
            "fluid": self.fluid_combo.currentText()
        }
        if self.fluid_combo.currentText() == "Crude Oil":
            data["api_gravity"] = self.api_input.text()
        return data


# Standalone test harness
if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = PipePage()
    window.show()
    sys.exit(app.exec())
