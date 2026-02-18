"""Developer launcher for the legacy Purge Wizard pages.

This app is intentionally lightweight: it helps you open each existing page in one
window so you can quickly see what still works while rebuilding the full wizard
flow around the newer simulation engine.
"""

import sys
from dataclasses import dataclass
from typing import Callable, List

from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
)

from page2_pipe import PipePage
from page3_profile import ProfilePage
from page4_purgesetup import PurgeSetupPage
from page5_simulationsetup import SimulationSetupPage, NitrogenSetupPage


@dataclass
class WizardSection:
    """Metadata for one page loaded into the development launcher."""

    title: str
    description: str
    factory: Callable[[], QWidget]


class WizardDevLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Purge Wizard Rebuild Launcher")
        self.resize(1300, 850)

        sections: List[WizardSection] = [
            WizardSection(
                title="1) Pipe & Fluid Inputs",
                description="Collect line geometry, roughness, and fluid properties.",
                factory=PipePage,
            ),
            WizardSection(
                title="2) Profile Loader",
                description="Load and validate profile files before purge setup.",
                factory=ProfilePage,
            ),
            WizardSection(
                title="3) Purge Setup",
                description="Define purge window and endpoint behavior.",
                factory=PurgeSetupPage,
            ),
            WizardSection(
                title="4) Simulation Setup",
                description="Configure pressure and speed constraints for run logic.",
                factory=SimulationSetupPage,
            ),
            WizardSection(
                title="5) Nitrogen Setup",
                description="Set N₂ supply assumptions and gas conditions.",
                factory=NitrogenSetupPage,
            ),
        ]

        root = QWidget()
        self.setCentralWidget(root)

        layout = QHBoxLayout(root)

        left = QVBoxLayout()
        self.page_list = QListWidget()
        self.page_list.currentRowChanged.connect(self.on_page_changed)
        left.addWidget(QLabel("Legacy Wizard Sections"))
        left.addWidget(self.page_list)

        self.open_help_btn = QPushButton("How to rebuild this wizard")
        self.open_help_btn.clicked.connect(self.show_rebuild_help)
        left.addWidget(self.open_help_btn)

        layout.addLayout(left, stretch=1)

        right = QVBoxLayout()
        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        right.addWidget(self.description_label)

        self.stack = QStackedWidget()
        right.addWidget(self.stack)

        layout.addLayout(right, stretch=4)

        self.sections = sections
        for section in self.sections:
            self.page_list.addItem(section.title)
            self.stack.addWidget(section.factory())

        self.page_list.setCurrentRow(0)

    def on_page_changed(self, index: int):
        if index < 0 or index >= len(self.sections):
            return
        self.stack.setCurrentIndex(index)
        self.description_label.setText(self.sections[index].description)

    def show_rebuild_help(self):
        QMessageBox.information(
            self,
            "Wizard rebuild checklist",
            "1) Make each page return a clean data dictionary.\n"
            "2) Store that dictionary in a shared state object.\n"
            "3) Connect Next/Back navigation with input validation.\n"
            "4) Move simulation math calls to page6_summary.py (engine module).\n"
            "5) Add one integration test path from page 1 to results export.",
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WizardDevLauncher()
    window.show()
    sys.exit(app.exec())
