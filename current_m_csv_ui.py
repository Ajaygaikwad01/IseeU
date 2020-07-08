import pandas as pd
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from pandas import DataFrame
import sys

from datetime import datetime
now = datetime.now()
current_month = now.strftime("%m")
current_year = now.strftime("%Y")
df = pd.read_csv(f'csv_dataset/({current_year}.{current_month})attendance.csv')

class pandasModel(QAbstractTableModel):

    def __init__(self,data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


class App(QDialog):
    def __init__(self):
        super().__init__()

        self.setGeometry(100, 100, 1080, 480)

        self.tableWidget = pandasModel(df)
        #self.model = pandasModel(df)
        self.view = QTableView()
        self.view.setModel(self.tableWidget)
        self.view.resize(800, 600)
        #self.view.show()
        #self.setCentralWidget(self.view)
       # self.tableWidget = self.view



        self.layout = QVBoxLayout()
        self.layout.addWidget(self.view)

        self.setLayout(self.layout)





if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())