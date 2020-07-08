import sys
import os
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import *
from PyQt5 import QtCore,QtGui,QtWidgets

class csvrecord_tab(QDialog):
    def __init__(self):
        super(csvrecord_tab,self).__init__()
        loadUi("CSV_RECORD.ui", self)
        self.Return = 0
        self.treeView.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.treeView.customContextMenuRequested.connect(self.context_menu)
        self.browse()

    def browse(self):
        path = "csv_dataset/"
        self.model = QtWidgets.QFileSystemModel()
        self.model.setRootPath((QtCore.QDir.rootPath()))
        self.treeView.setModel(self.model)
        self.treeView.setRootIndex(self.model.index(path))
        self.treeView.setSortingEnabled(True)

    def context_menu(self):
        menu = QtWidgets.QMenu()
        open = menu.addAction("Open")
        open.triggered.connect(self.open_file)
        delete = menu.addAction("Delete")
        delete.triggered.connect(self.delete_file)
        cursor = QtGui.QCursor()
        menu.exec_(cursor.pos())

    def open_file(self):
        index = self.treeView.currentIndex()
        file_path = self.model.filePath(index)
        os.startfile(file_path)

    def delete_file(self):
        index = self.treeView.currentIndex()
        file_path = self.model.filePath(index)
        os.remove(file_path)

if __name__=="__main__":
    app = QApplication(sys.argv)
    window = csvrecord_tab()
    window.show()
    try:
      sys.exit(app.exec_())
    except:
       print("exit")