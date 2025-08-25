import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel,
    QLineEdit, QCheckBox, QPushButton, QHBoxLayout, QRadioButton,
    QButtonGroup, QTableWidget, QTableWidgetItem, QGroupBox,
    QFormLayout, QScrollArea, QStackedWidget, QSlider
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator

# ----------------------------
# Simulation Setup
# ----------------------------
class SimulationSetupPage(QWidget):
    def __init__(self, purge_length="0", purge_volume_scf="0", start_mp="0", end_mp="0"):
        super().__init__()
        self._on_back = None
        self._on_next = None

        # Purge volume of the pipe section in **standard cubic feet** (scf @ 14.7 psia)
        try:
            self._purge_volume_scf = float(purge_volume_scf)
        except Exception:
            self._purge_volume_scf = 0.0

        outer = QVBoxLayout(self)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        outer.addWidget(self.scroll)

        content = QWidget()
        root = QVBoxLayout(content)
        root.setSpacing(3)
        root.setContentsMargins(5, 5, 5, 5)
        self.scroll.setWidget(content)

        title = QLabel("Simulation Setup")
        root.addWidget(title)

        self.purge_summary = QLabel(
            f"Purge MPs: {start_mp} - {end_mp} | "
            f"Length: {purge_length} miles | "
            f"Purge Volume: {purge_volume_scf} scf"
        )
        root.addWidget(self.purge_summary)

        speed_group = QGroupBox("Pig Speed Limits")
        speed_layout = QFormLayout()
        self.pig_min = QLineEdit(); self.pig_min.setValidator(QIntValidator(0, 9999, self))
        self.pig_max = QLineEdit(); self.pig_max.setValidator(QIntValidator(0, 9999, self))
        self.pig_min.setMaximumWidth(120)
        self.pig_max.setMaximumWidth(120)
        speed_layout.addRow("Min:", self.pig_min)
        speed_layout.addRow("Max:", self.pig_max)

        unit_layout = QHBoxLayout()
        self.unit_group = QButtonGroup(self)
        self.radio_mph = QRadioButton("mph")
        self.radio_fts = QRadioButton("ft/s")
        self.unit_group.addButton(self.radio_mph)
        self.unit_group.addButton(self.radio_fts)
        self.radio_mph.setChecked(True)
        unit_layout.addWidget(self.radio_mph)
        unit_layout.addWidget(self.radio_fts)

        speed_layout.addRow("Units:", unit_layout)
        speed_group.setLayout(speed_layout)
        root.addWidget(speed_group)

        pressure_group = QGroupBox("Maximum Allowable Pressure")
        pressure_layout = QFormLayout()
        self.max_pressure = QLineEdit(); self.max_pressure.setValidator(QIntValidator(0, 100000, self))
        self.max_pressure.setMaximumWidth(150)
        pressure_layout.addRow("Pressure (psi):", self.max_pressure)
        pressure_group.setLayout(pressure_layout)
        root.addWidget(pressure_group)

        constraints_group = QGroupBox("Hard Constraints")
        constraints_layout = QVBoxLayout()
        self.cap_speed_max = QCheckBox("Pig Speed Max")
        self.cap_speed_min = QCheckBox("Pig Speed Min")
        self.cap_drive_pressure = QCheckBox("Drive Pressure")
        self.cap_poi = QCheckBox("Pressure at POI's")
        self.cap_endpoint = QCheckBox("System Endpoint")
        for cb in [self.cap_speed_max, self.cap_speed_min, self.cap_drive_pressure, self.cap_poi, self.cap_endpoint]:
            constraints_layout.addWidget(cb)
        constraints_group.setLayout(constraints_layout)
        root.addWidget(constraints_group)

        self.poi_group = QGroupBox("Points of Interest (MP & Pressure)")
        poi_layout = QVBoxLayout()
        self.poi_table = QTableWidget(1, 2)
        self.poi_table.setHorizontalHeaderLabels(["MP", "Pressure (psi)"])
        poi_layout.addWidget(self.poi_table)
        poi_add_btn = QPushButton("Add POI")
        poi_add_btn.clicked.connect(self.add_poi_row)
        poi_layout.addWidget(poi_add_btn)
        self.poi_group.setLayout(poi_layout)
        root.addWidget(self.poi_group)

        self.endpoint_group = QGroupBox("System Endpoint Pressure")
        endpoint_layout = QFormLayout()
        self.endpoint_pressure = QLineEdit(); self.endpoint_pressure.setValidator(QIntValidator(0, 100000, self))
        self.endpoint_pressure.setMaximumWidth(150)
        endpoint_layout.addRow("Pressure (psi)", self.endpoint_pressure)
        self.endpoint_group.setLayout(endpoint_layout)
        root.addWidget(self.endpoint_group)

        self.poi_group.setHidden(True)
        self.endpoint_group.setHidden(True)

        self.cap_poi.toggled.connect(self.toggle_poi)
        self.cap_endpoint.toggled.connect(self.toggle_endpoint)

        nav_layout = QHBoxLayout()
        self.back_btn = QPushButton("Back")
        self.close_btn = QPushButton("Close")
        self.next_btn = QPushButton("Next")
        self.close_btn.clicked.connect(lambda: sys.exit(0))
        self.back_btn.clicked.connect(self._emit_back)
        self.next_btn.clicked.connect(self._emit_next)
        for b in [self.back_btn, self.close_btn, self.next_btn]:
            nav_layout.addWidget(b)
        outer.addLayout(nav_layout)

    def set_callbacks(self, on_back=None, on_next=None):
        self._on_back, self._on_next = on_back, on_next

    def _emit_back(self):
        if callable(self._on_back):
            self._on_back()

    def _emit_next(self):
        if callable(self._on_next):
            self._on_next(self.get_state())

    def add_poi_row(self):
        row = self.poi_table.rowCount()
        self.poi_table.insertRow(row)
        self.poi_table.setItem(row, 0, QTableWidgetItem(""))
        self.poi_table.setItem(row, 1, QTableWidgetItem(""))

    def toggle_poi(self, checked: bool):
        self.poi_group.setHidden(not checked)

    def toggle_endpoint(self, checked: bool):
        self.endpoint_group.setHidden(not checked)

    def get_state(self) -> dict:
        pois = []
        if not self.poi_group.isHidden():
            for r in range(self.poi_table.rowCount()):
                mp_item = self.poi_table.item(r, 0)
                p_item = self.poi_table.item(r, 1)
                if (mp_item and mp_item.text().strip()) or (p_item and p_item.text().strip()):
                    pois.append({"mp": mp_item.text().strip(), "pressure": p_item.text().strip()})

        return {
            "pig_min": self.pig_min.text().strip(),
            "pig_max": self.pig_max.text().strip(),
            "pig_units": "mph" if self.radio_mph.isChecked() else "ft/s",
            "max_pressure": self.max_pressure.text().strip(),
            "cap_speed_max": self.cap_speed_max.isChecked(),
            "cap_speed_min": self.cap_speed_min.isChecked(),
            "cap_drive_pressure": self.cap_drive_pressure.isChecked(),
            "cap_poi": self.cap_poi.isChecked(),
            "cap_endpoint": self.cap_endpoint.isChecked(),
            "pois": pois,
            "endpoint_pressure": self.endpoint_pressure.text().strip() if not self.endpoint_group.isHidden() else "",
            "purge_summary": self.purge_summary.text(),
            # pass purge volume forward (scf @ 14.7 psia)
            "purge_volume_scf": self._purge_volume_scf,
        }


# ----------------------------
# Stripped Nitrogen Setup Page
# ----------------------------
class NitrogenSetupPage(QWidget):
    def __init__(self):
        super().__init__()
        self._on_back = None
        self._on_next = None
        self.purge_volume_scf = 0.0

        outer = QVBoxLayout(self)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True)
        outer.addWidget(self.scroll)

        content = QWidget(); root = QVBoxLayout(content)
        self.scroll.setWidget(content)

        title = QLabel("Nitrogen Setup")
        title.setStyleSheet("font-weight:600;")
        root.addWidget(title)

        self.prev_summary = QLabel("")
        self.prev_summary.setWordWrap(True)
        root.addWidget(self.prev_summary)

        # Pressure slider and dynamic requirement readout
        cutoff_group = QGroupBox("Best guess for nitrogen pressure at end of purge (psig)")
        cutoff_v = QVBoxLayout(); cutoff_group.setLayout(cutoff_v)
        self.cutoff_slider = QSlider(Qt.Horizontal)
        self.cutoff_slider.setMinimum(0); self.cutoff_slider.setMaximum(2000)
        self.cutoff_slider.setSingleStep(50); self.cutoff_slider.setPageStep(50)
        self.cutoff_slider.setTickInterval(50); self.cutoff_slider.setTickPosition(QSlider.TicksBelow)
        self.cutoff_slider.setValue(500)
        self.cutoff_slider.valueChanged.connect(self._snap_and_update)

        self.cutoff_value = QLineEdit("500"); self.cutoff_value.setReadOnly(True)
        self.volume_readout = QLabel("Nitrogen required (scf @ 14.7 psia): 0")

        cutoff_v.addWidget(self.cutoff_slider)
        cutoff_v.addWidget(self.cutoff_value)
        cutoff_v.addWidget(self.volume_readout)
        root.addWidget(cutoff_group)

        nav = QHBoxLayout()
        self.back_btn = QPushButton("Back")
        self.close_btn = QPushButton("Close")
        self.next_btn = QPushButton("Next")
        self.close_btn.clicked.connect(lambda: sys.exit(0))
        self.back_btn.clicked.connect(lambda: self._on_back() if callable(self._on_back) else None)
        self.next_btn.clicked.connect(lambda: self._on_next(self.get_state()) if callable(self._on_next) else None)
        for b in [self.back_btn, self.close_btn, self.next_btn]:
            nav.addWidget(b)
        outer.addLayout(nav)

        # initialize display
        self._snap_and_update(self.cutoff_slider.value())

    def _compute_required_scf(self, target_psig: float) -> float:
        # Required nitrogen to raise pipe from 14.7 psia to (target_psig + 14.7) psia, isothermal.
        # n2_required (scf) = ((P2_abs / P1_abs) - 1) * purge_volume_scf
        P1_abs = 14.7  # psia
        P2_abs = float(target_psig) + 14.7
        if P2_abs <= P1_abs or self.purge_volume_scf <= 0:
            return 0.0
        return ((P2_abs / P1_abs) - 1.0) * float(self.purge_volume_scf)

    def _snap_and_update(self, v):
        snapped = round(v / 50) * 50
        self.cutoff_slider.blockSignals(True)
        self.cutoff_slider.setValue(snapped)
        self.cutoff_slider.blockSignals(False)
        self.cutoff_value.setText(str(snapped))

        required_scf = self._compute_required_scf(snapped)
        self.volume_readout.setText(f"Nitrogen required (scf @ 14.7 psia): {required_scf:,.0f}")

    def load_from_sim_state(self, sim_state: dict):
        self.prev_summary.setText(sim_state.get("purge_summary", ""))
        self.purge_volume_scf = sim_state.get("purge_volume_scf", 0.0)
        self._snap_and_update(self.cutoff_slider.value())

    def set_callbacks(self, on_back=None, on_next=None):
        self._on_back, self._on_next = on_back, on_next

    def get_state(self) -> dict:
        est_text = self.volume_readout.text().split(": ")[-1].replace(",", "")
        return {
            "target_pressure_psig": str(self.cutoff_slider.value()),
            "required_n2_scf": est_text,
        }


# ----------------------------
# Stacked Wizard Container
# ----------------------------
class SimulationWizard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulation Setup Wizard")
        self.resize(520, 640)
        self.setMinimumSize(self.size())

        self.stack = QStackedWidget(); self.setCentralWidget(self.stack)

        purge_length = "6.5"              # from purge setup packet
        purge_volume_scf = "10700"        # from pipe data packet (placeholder)
        start_mp = "0"; end_mp = "6.5"

        self.page_sim = SimulationSetupPage(purge_length, purge_volume_scf, start_mp, end_mp)
        self.page_sim.set_callbacks(on_back=self._noop_back, on_next=self._go_to_nitrogen)

        self.page_n2 = NitrogenSetupPage()
        self.page_n2.set_callbacks(on_back=self._back_to_sim, on_next=self._noop_next)

        self.stack.addWidget(self.page_sim)
        self.stack.addWidget(self.page_n2)
        self.stack.setCurrentIndex(0)

    def _noop_back(self):
        pass

    def _noop_next(self, *_):
        pass

    def _go_to_nitrogen(self, sim_state: dict):
        self.page_n2.load_from_sim_state(sim_state)
        self.stack.setCurrentIndex(1)

    def _back_to_sim(self):
        self.stack.setCurrentIndex(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SimulationWizard()
    w.show()
    sys.exit(app.exec())
