from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
from PyQt5.uic import loadUi
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
class add_csv(QDialog):
    def __init__(self):
        super(add_csv,self).__init__()
        loadUi("add_csv_ui.ui", self)
        self.Return = 0
        self.pushButton.clicked.connect(self.file_open)
    def file_open(self):
        try:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(QtWidgets.QDialog(), 'Open csv', QtCore.QDir.rootPath(),'*.csv')
            df1 = pd.read_csv(path)
            print(df1)
            df1.to_csv('main_data/info.csv')
        except:
            print("no file selected")
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = add_csv()
    ex.show()
    sys.exit(app.exec_())