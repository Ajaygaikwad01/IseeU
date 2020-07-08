import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5 import QtCore,QtGui,QtWidgets
import  os
class videorecord_tab(QDialog):
    def __init__(self):
        super(videorecord_tab,self).__init__()
        loadUi("video_Records.ui", self)
        self.Return = 0
        #self.pushButton4.clicked.connect(self.retrive4)
        self.listView.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.listView.customContextMenuRequested.connect(self.context_menu)
        self.browse()

    def browse(self):
        path = "video_dataset"
        self.model = QtWidgets.QFileSystemModel()
        self.model.setRootPath((QtCore.QDir.rootPath()))
        self.treeView.setModel(self.model)
        self.treeView.setRootIndex(self.model.index(path))
        self.treeView.setSortingEnabled(True)

        self.dirModel = QFileSystemModel()
        self.dirModel.setRootPath(QDir.rootPath())
        self.dirModel.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs)

        self.fileModel = QFileSystemModel()
        self.fileModel.setFilter(QDir.NoDotAndDotDot | QDir.Files)

        self.treeView.setModel(self.dirModel)
        self.listView.setModel(self.fileModel)

        self.treeView.setRootIndex(self.dirModel.index(path))
        self.listView.setRootIndex(self.fileModel.index(path))

        self.treeView.clicked.connect(self.on_clicked)
    def on_clicked(self, index):
        path = self.dirModel.fileInfo(index).absoluteFilePath()
        self.listView.setRootIndex(self.fileModel.setRootPath(path))

    def context_menu(self):
        menu = QtWidgets.QMenu()
        open = menu.addAction("Open")
        open.triggered.connect(self.open_file)
        delete = menu.addAction("Delete")
        delete.triggered.connect(self.delete_file)
        cursor = QtGui.QCursor()
        menu.exec_(cursor.pos())

    def open_file(self):
        index = self.listView.currentIndex()
        file_path = self.model.filePath(index)
        os.startfile(file_path)

    def delete_file(self):
        index = self.listView.currentIndex()
        file_path = self.model.filePath(index)
        os.remove(file_path)


if __name__=="__main__":
    app = QApplication(sys.argv)
    window = videorecord_tab()
    window.show()
    try:
      sys.exit(app.exec_())
    except:
       print("exit")