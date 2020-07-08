from PyQt5.QtWidgets import QDialog,QApplication,QMessageBox,QMainWindow
from PyQt5.uic import loadUi

class about(QDialog):
    def __init__(self):
        super(about,self).__init__()
        loadUi("About_ui.ui", self)
        self.Return = 0