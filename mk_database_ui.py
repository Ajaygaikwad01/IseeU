import sys
from PyQt5.QtCore import pyqtSlot

from PyQt5.QtWidgets import QDialog,QApplication,QMessageBox,QMainWindow
from PyQt5.uic import loadUi
from PyQt5.QtGui import QImage,QPixmap
from PyQt5 import QtCore,QtGui,QtWidgets
import pandas as pd
import cv2
import os

class mkdatabas(QDialog):
    def __init__(self):
        super(mkdatabas,self).__init__()
        loadUi("mk_photo_db.ui", self)
        self.Return = 0
        self.Button1.clicked.connect(self.take_photo)
        self.Button2.clicked.connect(self.train_db)

        self.treeView.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.treeView.customContextMenuRequested.connect(self.context_menu)
        self.browse()
        x = self.combobox1()
        y=self.combobox2()

    def combobox1(self):
        for i in range(0, 5):
            self.comboBox1.addItem(str(i))

    def combobox2(self):
        list=[30,50,70,80,100]
        for i in list:
            self.comboBox2.addItem(str(i))

    def show_popup(self, data):
        msg = QMessageBox()
        msg.setWindowTitle("Warning")
        msg.setText(data)
        msg.setIcon(QMessageBox.Warning)
        x = msg.exec_()

    def browse(self):
        path = "dataset"
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

    @pyqtSlot()
    def train_db(self):
        import numpy as np
        from PIL import Image
        path = 'dataset'
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier("main_data/haarcascade_frontalface_default.xml");
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []
        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
            img_numpy = np.array(PIL_img, 'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)

        #return faceSamples, ids
        pop_msg1 = "\n [INFO] Training faces. It will take a few seconds. Wait ..."
        self.show_popup(pop_msg1)
        #print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        #faces, ids = getImagesAndLabels(path)
        recognizer.train(faceSamples, np.array(ids))
    # Save the model into trainer/trainer.yml
        recognizer.save('main_data/trainer.yml')  # recognizer.save() worked on Mac, but not on Pi

        #print("\n [INFO] Training faces. It will take a few seconds. Wait ...")


        pop_msg="\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids)))
        self.show_popup(pop_msg)

    @pyqtSlot()
    def take_photo(self):
        face_cascade = cv2.CascadeClassifier('main_data/haarcascade_frontalface_default.xml')
        df = pd.read_csv('main_data/info.csv')
        df_id = df["ID"].values.tolist()

        port_id = self.comboBox1.currentText()
        cap = cv2.VideoCapture(int(port_id))

        try:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
        except:
            print("canot change camera size")
        try:


            face_id = self.plainTextEdit.toPlainText()
            face_name = self.plainTextEdit2.toPlainText()
            count = 0
            while (True):
                ret, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
                for (x, y, w, h) in faces:
                    print(x, y, w, h)
                    roi_gray = gray[y:y + h, x:x + h]
                    roi_color = frame[y:y + h, x:x + h]
                    count += 1
                    if face_id in str(df_id):
                        cv2.imwrite(f"dataset/{face_name}." + f"{face_id}.{count}.jpg", roi_gray)

                    else:

                        break
                        print("invalid data")


                    color = (96, 96, 0)  # BGR
                    stroke = 1
                    end_cordinate_x = x + w
                    end_cordinate_y = y + h
                    cv2.rectangle(frame, (x, y), (end_cordinate_x, end_cordinate_y), color, stroke)
                # cv2.imshow('FRAME', frame)
                self.displayImage(frame)

                count_id = self.comboBox2.currentText()
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
                if count >= int(count_id):
                    break



        except:

            popmsg4 = "camera not found"
            self.show_popup(popmsg4)

        cap.release()

        #
    def displayImage(self, img, window=True):
        qformat = QtGui.QImage.Format_Indexed8
        if len(img.shape)==3 :
            if img.shape[2]==4:
                qformat = QtGui.QImage.Format_RGBA8888
            else:
                qformat = QtGui.QImage.Format_RGB888
        outImage = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        if window:
            self.image.setPixmap(QtGui.QPixmap.fromImage(outImage))

if __name__=="__main__":
    app = QApplication(sys.argv)
    window = mkdatabas()
    window.show()
    try:
      sys.exit(app.exec_())
      cv2.destoryAllWindows()
    except:
       print("exit")