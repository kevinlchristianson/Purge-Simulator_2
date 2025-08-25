import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget

# Import your wizard pages (1–6)
from page1 import Page1
from page2 import Page2
from page3 import Page3
from page4 import Page4
from page5 import Page5
from page6 import Page6

# Shared state object to hold all inputs
class WizardState:
    def __init__(self):
        self.data = {}

class WizardController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pipeline Purging Simulator Wizard")
        self.resize(1200, 800)

        # Shared state
        self.state = WizardState()

        # Central stacked widget for wizard pages
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Initialize pages
        self.pages = [
            Page1(self.state, self.next_page, self.prev_page),
            Page2(self.state, self.next_page, self.prev_page),
            Page3(self.state, self.next_page, self.prev_page),
            Page4(self.state, self.next_page, self.prev_page),
            Page5(self.state, self.next_page, self.prev_page),
            Page6(self.state, self.next_page, self.prev_page),
        ]

        # Add to stack
        for page in self.pages:
            self.stack.addWidget(page)

        # Start on page 1
        self.stack.setCurrentIndex(0)

    def next_page(self):
        current = self.stack.currentIndex()
        if current < len(self.pages) - 1:
            self.stack.setCurrentIndex(current + 1)

    def prev_page(self):
        current = self.stack.currentIndex()
        if current > 0:
            self.stack.setCurrentIndex(current - 1)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WizardController()
    window.show()
    sys.exit(app.exec())
