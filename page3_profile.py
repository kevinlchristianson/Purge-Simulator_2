import os, re, zipfile, math
import xml.etree.ElementTree as ET
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QListWidget,
    QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem, QHBoxLayout
)


# -----------------------------
# Helpers
# -----------------------------
def haversine_miles(lat1, lon1, lat2, lon2):
    if any(pd.isna([lat1, lon1, lat2, lon2])): return 0.0
    R = 3958.7613
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * (2*math.atan2(math.sqrt(a), math.sqrt(1-a)))

def normalize_columns(df):
    return {c.lower().strip(): c for c in df.columns.tolist()}

def find_first_col(norm_map, keys):
    for k in norm_map:
        for needle in keys:
            if needle in k:
                return norm_map[k]
    return None


class ProfilePage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Load Profile Files (.kmz, .kml, .xlsx, .txt, .csv)"))

        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.show_errors)

        # Buttons row
        btn_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Profile")
        self.load_button.clicked.connect(self.load_file)
        self.replace_button = QPushButton("Replace with Fixed Profile")
        self.replace_button.clicked.connect(self.replace_file)
        self.remove_button = QPushButton("Remove Selected Profile")
        self.remove_button.clicked.connect(self.remove_file)
        self.summary_button = QPushButton("Show Summary")
        self.summary_button.clicked.connect(self.show_summary)
        self.export_button = QPushButton("Export Error Report")
        self.export_button.clicked.connect(self.export_errors)

        btn_layout.addWidget(self.load_button)
        btn_layout.addWidget(self.replace_button)
        btn_layout.addWidget(self.remove_button)

        layout.addWidget(self.file_list)
        layout.addLayout(btn_layout)
        layout.addWidget(self.summary_button)
        layout.addWidget(self.export_button)

        self.link_label = QLabel(
            '<a href="https://www.gpsvisualizer.com/elevation">Missing or Invalid elevation? '
            'Get Elevation Data Here</a>'
        )
        self.link_label.setOpenExternalLinks(True)
        self.link_label.hide()
        layout.addWidget(self.link_label)

        self.setLayout(layout)
        self.segment_data = []    # (filename, length, points, status)
        self.error_map = {}       # base -> errors

    # -----------------------------
    # File management
    # -----------------------------
    def load_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Profile File", "", 
                                                   "Profile Files (*.kmz *.kml *.xlsx *.txt *.csv)")
        if file_name:
            self._process_file(file_name)

    def replace_file(self):
        selected = self.file_list.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Replace Profile", "Select a profile to replace.")
            return
        old_item = selected[0]
        idx = self.file_list.row(old_item)
        new_file, _ = QFileDialog.getOpenFileName(self, "Replace Profile File", "", 
                                                  "Profile Files (*.kmz *.kml *.xlsx *.txt *.csv)")
        if new_file:
            self._process_file(new_file, replace_idx=idx)

    def remove_file(self):
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Remove Profile", "No profile selected.")
            return
        for item in selected_items:
            row = self.file_list.row(item)
            self.file_list.takeItem(row)
            if row < len(self.segment_data):
                self.segment_data.pop(row)

    # -----------------------------
    # Process & evaluate
    # -----------------------------
    def _process_file(self, file_name, replace_idx=None):
        needs_fix, msg, errors, length_miles, total_points = self.check_profile(file_name)
        base = os.path.basename(file_name)
        self.error_map[base] = errors

        status_icon = "⚠" if needs_fix else "✔"
        item_text = f"{base} {status_icon} {msg} | {length_miles:.2f} mi | {total_points} pts"

        if replace_idx is not None:
            self.file_list.takeItem(replace_idx)
            self.segment_data[replace_idx] = (file_name, length_miles, total_points, status_icon)
            self.file_list.insertItem(replace_idx, item_text)
        else:
            self.segment_data.append((file_name, length_miles, total_points, status_icon))
            self.file_list.addItem(item_text)

        if needs_fix:
            self.link_label.show()

    def show_errors(self, item):
        base = item.text().split()[0]
        if base in self.error_map and self.error_map[base]:
            QMessageBox.warning(self, f"Issues in {base}", "\n".join(self.error_map[base]))

    # -----------------------------
    # Summary & error export
    # -----------------------------
    def show_summary(self):
        if not self.segment_data:
            QMessageBox.information(self, "Summary", "No profiles loaded.")
            return

        table = QTableWidget()
        table.setRowCount(len(self.segment_data))
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Profile", "Status", "Length (mi)", "Points"])

        for i, (fname, length, pts, status) in enumerate(self.segment_data):
            table.setItem(i, 0, QTableWidgetItem(os.path.basename(fname)))
            table.setItem(i, 1, QTableWidgetItem(status))
            table.setItem(i, 2, QTableWidgetItem(f"{length:.2f}"))
            table.setItem(i, 3, QTableWidgetItem(str(pts)))

        msg_box = QMessageBox()
        msg_box.setWindowTitle("Profiles Summary")
        msg_box.setText("Loaded profiles overview:")
        msg_box.layout().addWidget(table)
        msg_box.exec()

    def export_errors(self):
        if not self.error_map:
            QMessageBox.information(self, "Export Errors", "No errors to export.")
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Error Report", "profile_errors.csv", "CSV Files (*.csv)"
        )
        if save_path:
            with open(save_path, 'w') as f:
                f.write("Profile,Error\n")
                for base, errs in self.error_map.items():
                    for e in errs:
                        f.write(f"{base},{e}\n")
            QMessageBox.information(self, "Export Complete", f"Errors saved to {save_path}")

    # -----------------------------
    # Check profile (full implementation)
    # -----------------------------
    def check_profile(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".kmz", ".kml"):
            return self._check_kmz_kml(file_path)
        elif ext in (".xlsx", ".xls"):
            return self._check_table(file_path, kind="excel")
        elif ext in (".txt", ".csv"):
            delim = '\t' if ext == ".txt" else ','
            return self._check_table(file_path, kind="text", delimiter=delim)
        else:
            return True, "Unsupported format", ["Unsupported format"], 0.0, 0

    def _check_kmz_kml(self, file_path):
        try:
            if file_path.endswith('.kmz'):
                with zipfile.ZipFile(file_path, 'r') as z:
                    kml_names = [n for n in z.namelist() if n.endswith('.kml')]
                    if not kml_names:
                        return True, "No KML found inside KMZ", ["missing kml"], 0.0, 0
                    data = z.read(kml_names[0])
            else:
                with open(file_path, 'rb') as f:
                    data = f.read()

            root = ET.fromstring(data)

            coords = []
            for elem in root.iter():
                if elem.tag.lower().endswith('coordinates') and elem.text:
                    for chunk in elem.text.strip().split():
                        parts = chunk.split(',')
                        if len(parts) >= 2:
                            lon = float(parts[0])
                            lat = float(parts[1])
                            elev = float(parts[2]) if len(parts) >= 3 and parts[2] != '' else float('nan')
                            coords.append((lon, lat, elev))

            total_points = len(coords)
            if total_points < 2:
                return True, "Too few points to evaluate length", ["insufficient points"], 0.0, total_points

            elevs = pd.Series([c[2] for c in coords], dtype='float64')
            zero_count = (elevs == 0).sum()
            missing_count = elevs.isna().sum()

            bad_idx = []
            for i in range(1, total_points):
                e1, e2 = elevs.iloc[i-1], elevs.iloc[i]
                if not (pd.isna(e1) or pd.isna(e2)) and abs(e2 - e1) > 1000:
                    bad_idx.append(i)

            length_miles = 0.0
            for i in range(1, total_points):
                length_miles += haversine_miles(coords[i-1][1], coords[i-1][0], coords[i][1], coords[i][0])

            errors = []
            needs_fix = False
            if missing_count > 0:
                needs_fix = True
                errors.append(f"Missing elevation data: {missing_count} points")
            if zero_count > 0:
                needs_fix = True
                errors.append(f"Elevation points with 0: {zero_count} points")
            if bad_idx:
                needs_fix = True
                errors.append(f"Erroneous elevation jumps: {len(bad_idx)} points")

            if needs_fix:
                msg = f"Issues found — Total {total_points}, Missing {missing_count}, Zeros {zero_count}, Erroneous {len(bad_idx)}"
                return True, msg, errors, length_miles, total_points
            else:
                return False, "OK", [], length_miles, total_points
        except Exception as e:
            return True, f"Error reading profile: {e}", [str(e)], 0.0, 0

    def _check_table(self, file_path, kind="text", delimiter='\t'):
        try:
            if kind == "excel":
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path, sep=delimiter, engine='python')

            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype(str).apply(lambda x: re.sub(r"<.*?>", "", x))

            norm = normalize_columns(df)
            elev_col = find_first_col(norm, ['elev', 'altitude', 'alt (ft)', 'alt ', ' alt', ' z', '(ft)', 'height'])
            lat_col = find_first_col(norm, ['latitude', 'lat'])
            lon_col = find_first_col(norm, ['longitude', 'lon', 'long'])
            mp_col = find_first_col(norm, ['milepost', 'mile post', 'mp'])

            if elev_col is None:
                return True, f"No elevation column found. Columns: {list(df.columns)}", ["No elevation column"], 0.0, 0

            df[elev_col] = pd.to_numeric(df[elev_col], errors='coerce')
            zero_count = (df[elev_col] == 0).sum()
            missing_count = df[elev_col].isna().sum()

            total_points = len(df)

            bad_idx = []
            if total_points > 1:
                e = df[elev_col]
                for i in range(1, total_points):
                    e1, e2 = e.iloc[i-1], e.iloc[i]
                    if not (pd.isna(e1) or pd.isna(e2)) and abs(e2 - e1) > 1000:
                        bad_idx.append(i)

            length_miles = 0.0
            if lat_col and lon_col:
                df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
                df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
                for i in range(1, total_points):
                    length_miles += haversine_miles(
                        df[lat_col].iloc[i-1], df[lon_col].iloc[i-1], df[lat_col].iloc[i], df[lon_col].iloc[i]
                    )
            elif mp_col:
                df[mp_col] = pd.to_numeric(df[mp_col], errors='coerce')
                length_miles = float(df[mp_col].max() - df[mp_col].min())
            else:
                return True, "Cannot compute length — need Latitude/Longitude or Milepost column.", ["missing distance basis"], 0.0, total_points

            errors = []
            needs_fix = False
            if missing_count > 0:
                needs_fix = True
                errors.append(f"Missing elevation values: {missing_count}")
            if zero_count > 0:
                needs_fix = True
                errors.append(f"Elevation zeros: {zero_count}")
            if bad_idx:
                needs_fix = True
                errors.append(f"Erroneous elevation jumps: {len(bad_idx)}")

            if needs_fix:
                msg = f"Issues found — Total {total_points}, Missing {missing_count}, Zeros {zero_count}, Erroneous {len(bad_idx)}"
                return True, msg, errors, length_miles, total_points
            else:
                return False, "OK", [], length_miles, total_points
        except Exception as e:
            return True, f"Error reading profile: {e}", [str(e)], 0.0, 0


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = ProfilePage()
    window.setWindowTitle("Profile Page Debug")
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())
