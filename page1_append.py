import sys, os
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QListWidget,
    QPushButton, QMessageBox, QTableWidget, QTableWidgetItem,
    QFileDialog, QStackedWidget, QHBoxLayout
)
from profile_page import get_loaded_segments  # assumes this provides segment_data
from purgesetup_page import PurgeSetupPage   # target page

class AppendPage(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.segment_data = get_loaded_segments()
        self.stack = stack
        self.selection_order = []
        layout = QVBoxLayout()

        instruction_label = QLabel("Click the segments in the desired order. Numbers will appear to show sequence.")
        layout.addWidget(instruction_label)

        self.segment_list = QListWidget()
        self.segment_list.itemClicked.connect(self.handle_selection)

        # Buttons
        self.confirm_button = QPushButton("Confirm Order")
        self.confirm_button.clicked.connect(self.confirm_order)
        self.confirm_button.setEnabled(False)

        self.reset_button = QPushButton("Reset Selection")
        self.reset_button.clicked.connect(self.reset_selection)
        self.reset_button.setEnabled(False)

        self.next_button = QPushButton("Append && Show Summary")
        self.next_button.clicked.connect(self.show_summary)
        self.next_button.setEnabled(False)

        self.export_button = QPushButton("Export Appended Profile")
        self.export_button.clicked.connect(self.export_profile)
        self.export_button.setEnabled(False)

        layout.addWidget(self.segment_list)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.confirm_button)
        button_layout.addWidget(self.reset_button)
        layout.addLayout(button_layout)

        layout.addWidget(self.next_button)
        layout.addWidget(self.export_button)

        self.setLayout(layout)
        self.appended_segments = []

        # preload list
        for f, length, pts in self.segment_data:
            self.segment_list.addItem(f"{os.path.basename(f)} | {length:.2f} miles | {pts} points")

    def handle_selection(self, item):
        idx = self.segment_list.row(item)
        if idx not in self.selection_order:
            self.selection_order.append(idx)
        else:
            self.selection_order.remove(idx)
        self.update_labels()
        self.confirm_button.setEnabled(bool(self.selection_order))
        self.reset_button.setEnabled(bool(self.selection_order))

    def update_labels(self):
        for i in range(self.segment_list.count()):
            base_text = self.segment_data[i][0].split(os.sep)[-1]
            length, pts = self.segment_data[i][1], self.segment_data[i][2]
            label = f"{base_text} | {length:.2f} miles | {pts} points"
            if i in self.selection_order:
                order_num = self.selection_order.index(i) + 1
                label = f"[{order_num}] {label}"
            self.segment_list.item(i).setText(label)

    def reset_selection(self):
        self.selection_order.clear()
        self.update_labels()
        self.appended_segments = []
        self.confirm_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.export_button.setEnabled(False)

    def confirm_order(self):
        if not self.selection_order:
            QMessageBox.warning(self, "Confirm Order", "No segments selected.")
            return

        self.appended_segments = [self.segment_data[i] for i in self.selection_order]
        self.next_button.setEnabled(True)
        self.export_button.setEnabled(True)

    def show_summary(self):
        if not self.appended_segments:
            QMessageBox.warning(self, "Summary", "No segments confirmed.")
            return

        total_length = sum(seg[1] for seg in self.appended_segments)
        total_points = sum(seg[2] for seg in self.appended_segments)

        summary_page = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Appended Profile Summary"))

        table = QTableWidget()
        table.setRowCount(len(self.appended_segments))
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Segment", "Length (miles)", "Points"])

        for i, (fname, length, pts) in enumerate(self.appended_segments):
            table.setItem(i, 0, QTableWidgetItem(os.path.basename(fname)))
            table.setItem(i, 1, QTableWidgetItem(f"{length:.2f}"))
            table.setItem(i, 2, QTableWidgetItem(str(pts)))

        layout.addWidget(table)
        layout.addWidget(QLabel(f"Total Length: {total_length:.2f} miles"))
        layout.addWidget(QLabel(f"Total Points: {total_points}"))

        # Buttons on summary page
        button_layout = QHBoxLayout()
        proceed_button = QPushButton("Confirm && Proceed to Purge Setup")
        back_button = QPushButton("Back to Append Page")
        button_layout.addWidget(proceed_button)
        button_layout.addWidget(back_button)

        layout.addLayout(button_layout)

        summary_page.setLayout(layout)
        self.stack.addWidget(summary_page)
        self.stack.setCurrentWidget(summary_page)

        back_button.clicked.connect(lambda: self.stack.setCurrentWidget(self))
        proceed_button.clicked.connect(self.go_to_purge)

    def export_profile(self):
        if not self.appended_segments:
            QMessageBox.warning(self, "Export Profile", "No appended profile confirmed.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Save")
        if not folder:
            return
        save_path = os.path.join(folder, "appended_profile.txt")
        with open(save_path, 'w') as f:
            for fname, length, pts in self.appended_segments:
                f.write(f"{fname},{length},{pts}\n")
        QMessageBox.information(self, "Export Complete", f"Appended profile saved to {save_path}")

    def go_to_purge(self):
        purge_page = PurgeSetupPage(self.appended_segments, self.stack)
        self.stack.addWidget(purge_page)
        self.stack.setCurrentWidget(purge_page)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    stack = QStackedWidget()
    window = AppendPage(stack)
    stack.addWidget(window)
    stack.show()
    sys.exit(app.exec())
