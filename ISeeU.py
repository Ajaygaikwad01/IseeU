import sys
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QApplication
from PyQt5.uic import loadUi
import cv2

from PyQt5 import QtGui
import numpy as np
import os
import time
import pandas as pd
from datetime import datetime
from random import randint
from playsound import playsound

beepsound="main_data/beep.mp3"

class MyPerson:
    tracks = []

    def __init__(self, i, xi, yi, max_age):
        self.i = i
        self.x = xi
        self.y = yi
        self.tracks = []
        self.R = randint(0, 255)
        self.G = randint(0, 255)
        self.B = randint(0, 255)
        self.done = False
        self.state = '0'
        self.age = 0
        self.max_age = max_age
        self.dir = None

    def getRGB(self):
        return (self.R, self.G, self.B)

    def getTracks(self):
        return self.tracks

    def getId(self):
        return self.i

    def getState(self):
        return self.state

    def getDir(self):
        return self.dir

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def updateCoords(self, xn, yn):
        self.age = 0
        self.tracks.append([self.x, self.y])
        self.x = xn
        self.y = yn

    def setDone(self):
        self.done = True

    def timedOut(self):
        return self.done

    def going_UP(self, mid_start, mid_end):
        if len(self.tracks) >= 2:
            if self.state == '0':

                if self.tracks[-1][1] < mid_end and self.tracks[-2][1] >= mid_end:  # cruzo la linea
                    state = '1'
                    self.dir = 'up'

                    return True
            else:
                return False
        else:
            return False

    def going_DOWN(self, mid_start, mid_end):
        if len(self.tracks) >= 2:
            if self.state == '0':
                if self.tracks[-1][1] > mid_start and self.tracks[-2][1] <= mid_start:  # cruzo la linea
                    state = '1'
                    self.dir = 'down'
                    return True
            else:
                return False
        else:
            return False

    def age_one(self):
        self.age += 1
        if self.age > self.max_age:
            self.done = True
        return True


class MultiPerson:
    def __init__(self, persons, xi, yi):
        self.persons = persons
        self.x = xi
        self.y = yi
        self.tracks = []
        self.R = randint(0, 255)
        self.G = randint(0, 255)
        self.B = randint(0, 255)
        self.done = False


class Window1(QMainWindow):
    def __init__(self):
        super(Window1, self).__init__()
        loadUi("icumain_ui.ui", self)
        self.Return = 0
        self.pushButton.clicked.connect(self.face_datection)
        self.resetButton.clicked.connect(self.count_reset)
        x = self.combobox1()
        y = self.combobox2()
        w = self.combobox3()
        z = self.combobox4()

        self.actionFor_face_Detection.triggered.connect(self.mk_db)
        self.actionAdd_csv_file.triggered.connect(self.addcsv)
        self.actioncurrent_csv.triggered.connect(self.current_csv)
        self.actionshow_Csv.triggered.connect(self.all_csv)
        self.actionshow_Videos.triggered.connect(self.video_records)
        self.actionAbout_Software.triggered.connect(self.about)

    def mk_db(self):
        from mk_database_ui import mkdatabas
        tab = mkdatabas()
        tab.exec_()

    def addcsv(self):
        from Addcsv_ui import add_csv
        tab = add_csv()
        tab.exec_()

    def current_csv(self):
        from current_m_csv_ui import App
        tab = App()
        tab.exec_()

    def all_csv(self):
        from csvrecord_ui import csvrecord_tab
        tab = csvrecord_tab()
        tab.exec_()

    def video_records(self):
        from videorecords_ui import videorecord_tab
        tab = videorecord_tab()
        tab.exec_()

    def about(self):
        from About_ui import about
        tab = about()
        tab.exec_()
    def combobox1(self):
        for i in range(0, 16):
            self.comboBox1.addItem(str(i))

    def combobox2(self):
        for i in range(0, 15):
            self.comboBox2.addItem(str(i))

    def combobox3(self):
        list = ['30 sec', '1 min', '1 Hour', '2 Hour', '1 Day']
        for i in list:
            self.comboBox3.addItem(i)

    def combobox4(self):
        list = ['30 sec', '1 min', '1 Hour', '2 Hour', '1 Day']
        for i in list:
            self.comboBox4.addItem(i)

    def show_popup(self, data):
        msg = QMessageBox()
        msg.setWindowTitle("Warning")
        msg.setText(data)
        msg.setIcon(QMessageBox.Warning)
        x = msg.exec_()

    def count_reset(self):
        self.cnt_down=0
        self.cnt_up=0





    def face_datection(self):


        if (self.comboBox3.currentText()) == '30 sec':
            duration1=30
        if (self.comboBox3.currentText()) == '1 min':
            duration1=60
        if (self.comboBox3.currentText()) == '1 Hour':
            duration1=3600
        if (self.comboBox3.currentText()) == '2 Hour':
            duration1=7200
        if (self.comboBox3.currentText()) == '1 Day':
            duration1=12960000

        if (self.comboBox4.currentText()) == '30 sec':
            duration2=30
        if (self.comboBox4.currentText()) == '1 min':
            duration2=60
        if (self.comboBox4.currentText()) == '1 Hour':
            duration2=3600
        if (self.comboBox4.currentText()) == '2 Hour':
            duration2=7200
        if (self.comboBox4.currentText()) == '1 Day':
            duration2=12960000

        #duration1 = int(self.comboBox3.currentText())
        #duration2 = int(self.comboBox4.currentText())
        print(duration1)
        print(duration2)

        port_id1 = self.comboBox1.currentText()
        port_id2 = self.comboBox2.currentText()

        if port_id1 == port_id2:
            cam = cv2.VideoCapture(0)
            cam2= cv2.VideoCapture(1)
            x = " cameras not working on same port"
            self.show_popup(x)

        else:
            cam = cv2.VideoCapture(int(port_id1))
            cam2 = cv2.VideoCapture(int(port_id2))

        self.cnt_up = 0
        self.cnt_down = 0
       # for i in range(19):
           # print(i, cam2.get(i))
        h = 480
        w = 640
        frameArea = h * w
        areaTH = frameArea / 250

        # Lineas de entrada/salida
        line_up = int(2 * (h / 5))
        line_down = int(3 * (h / 5))

        up_limit = int(1 * (h / 5))
        down_limit = int(4 * (h / 5))

        line_down_color = (255, 0, 0)
        line_up_color = (0, 0, 255)
        pt1 = [0, line_down]
        pt2 = [w, line_down]
        pts_L1 = np.array([pt1, pt2], np.int32)
        pts_L1 = pts_L1.reshape((-1, 1, 2))
        pt3 = [0, line_up]
        pt4 = [w, line_up]
        pts_L2 = np.array([pt3, pt4], np.int32)
        pts_L2 = pts_L2.reshape((-1, 1, 2))

        pt5 = [0, up_limit]
        pt6 = [w, up_limit]
        pts_L3 = np.array([pt5, pt6], np.int32)
        pts_L3 = pts_L3.reshape((-1, 1, 2))
        pt7 = [0, down_limit]
        pt8 = [w, down_limit]
        pts_L4 = np.array([pt7, pt8], np.int32)
        pts_L4 = pts_L4.reshape((-1, 1, 2))

        fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

        kernelOp = np.ones((3, 3), np.uint8)
        # kernelOp2 = np.ones((5, 5), np.uint8)
        kernelCl = np.ones((11, 11), np.uint8)

        # Variables
        font = cv2.FONT_HERSHEY_SIMPLEX
        persons = []
        max_p_age = 5
        pid = 1
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        now = datetime.now()
        current_Date = now.strftime("%d.%m.%Y")
        current_time = now.strftime("%H.%M.%S")
        start = time.time()
        if os.path.isdir(f"video_dataset/camera1/{current_Date}_VIDEO"):
            pass
        else:
            os.mkdir(f'video_dataset/camera1/{current_Date}_VIDEO')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if self.checkBox2.isChecked():
            video_writer = cv2.VideoWriter(f'video_dataset/camera1/{current_Date}_VIDEO/{current_time}.avi', fourcc, 20.0,
                                       (640, 480))

        if os.path.isdir(f"video_dataset/camera2/{current_Date}_VIDEO"):
            pass
        else:
            os.mkdir(f'video_dataset/camera2/{current_Date}_VIDEO')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if self.checkBox3.isChecked():
            video_writer2 = cv2.VideoWriter(f'video_dataset/camera2/{current_Date}_VIDEO/{current_time}.avi', fourcc, 20.0,
                                       (640, 480))
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        # port_id1 = self.comboBox1.currentText()

        # cam = cv2.VideoCapture(int(port_id1))

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('main_data/trainer.yml')
        faceCascade = cv2.CascadeClassifier("main_data/haarcascade_frontalface_default.xml");
        # iniciate id counter
        id = 0
        # getting values from database@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        df1 = pd.read_csv('main_data/info.csv', index_col="ID")
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        while True:
            self.listWidget.clear()
            self.listWidget2.clear()

            now = datetime.now()
            current_sec = now.strftime("%S")
            current_min = now.strftime("%M")
            current_hour = now.strftime("%H")
            current_time = now.strftime("%H.%M.%S")
            current_month = now.strftime("%m")
            current_day = now.strftime("%d")
            current_year = now.strftime("%Y")
            current_Date = now.strftime("%d.%m.%Y")
            # CREATE DATE VISE FOLDER @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            if os.path.isdir(f"video_dataset/camera1/{current_Date}_VIDEO"):
                pass
            else:
                os.mkdir(f'video_dataset/camera1/{current_Date}_VIDEO')

            if os.path.isdir(f"video_dataset/camera2/{current_Date}_VIDEO"):
                pass
            else:
                os.mkdir(f'video_dataset/camera2/{current_Date}_VIDEO')
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            try:
                ret, frame = cam.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y + h, x:x + h]

                    # Check if confidence is less them 100 ==> "0" is perfect match@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                    if (confidence < 100):
                        name_id = df1.loc[id, "Name"]
                        ocuupation_id = id
                        confidence = "  {0}%".format(round(100 - confidence))
                    else:
                        name_id = "unknown"
                        ocuupation_id = ""
                        id = 0
                        confidence = "  {0}%".format(round(100 - confidence))

                        if self.checkBox.isChecked():
                            playsound(beepsound)
                    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    # for saving attandance in database@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    if os.path.isfile(f'csv_dataset/({current_year}.{current_month})attendance.csv'):
                        df1 = pd.read_csv(f'csv_dataset/({current_year}.{current_month})attendance.csv', index_col="ID")
                    else:
                        df1 = pd.read_csv('main_data/info.csv', index_col="ID")
                    if current_Date not in df1.keys():
                        df1[current_Date] = "None"
                    if df1.loc[id, f"{current_Date}"] == "None":
                        df1.loc[id, f"{current_Date}"] = f"{current_hour}.{current_min}"
                    df1.to_csv(f'csv_dataset/({current_year}.{current_month})attendance.csv')
                    # df.to_csv('attendance.csv')
                    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    # for saving image with id, date,time
                    cv2.imwrite(
                        f"photo_dataset/{id}-{current_day}{current_month}" + f"{current_year}_{current_hour}{current_min}{current_sec}.jpg",
                        roi_gray)
                    # FOR DISPLAY DATA IN FRAME @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    color = (9, 255, 0)  # BGR
                    stroke = 1
                    end_cordinate_x = x + w
                    end_cordinate_y = y + h
                    cv2.rectangle(frame, (x, y), (end_cordinate_x, end_cordinate_y), color, stroke)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    white_color = (255, 255, 255)  # BGR
                    green_blue_color = (255, 255, 0)  # BGR
                    stroke = 1
                    font_size = 0.5
                    if (len(faces) == 1):
                        cv2.putText(frame, "TOTAL=1", (500, 40), font, font_size, white_color, stroke)
                    else:
                        cv2.putText(frame, "Total=" + str(len(faces)), (500, 40), font, font_size, white_color, stroke)
                    cv2.putText(frame, str(name_id), (x + 25, y - 25), font, font_size, green_blue_color, stroke)
                    cv2.putText(frame, str(ocuupation_id), (x + 35, y - 5), font, font_size, white_color, stroke)
                    cv2.putText(frame, str(confidence), (x + 5, y + h - 5), font, font_size, green_blue_color, stroke)

                    #self.textEdit4.setText(str(name_id))
                    self.listWidget2.addItem(str(name_id))
                    self.listWidget.addItem(str(id))
                font = cv2.FONT_HERSHEY_SIMPLEX
                white_color = (255, 255, 255)  # BGR
                green_blue_color = (255, 255, 0)  # BGR
                stroke = 1
                font_size = 0.5
                cv2.putText(frame, f"TIME:{current_time}", (300, 30), font, font_size,
                            white_color, stroke)
                cv2.putText(frame, f"DATE:{current_Date}", (150, 30), font, font_size,
                            white_color, stroke)
                # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                self.textEdit1.setText(current_Date)
                self.textEdit2.setText(current_time)
                #self.textEdit3.setText(str(id))

                # self.textEdit4.setText(str(name_id))
                self.textEdit5.setText(str(len(faces)))
                # cv2.imshow('camera', frame)
                self.displayImage(frame)
                # FOR SAVE AS VIDEO @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                if time.time() - start > duration1:
                    now = datetime.now()
                    # current_time = now.strftime("%H.%M.%S")

                    start = time.time()
                    if self.checkBox2.isChecked():
                       video_writer = cv2.VideoWriter(f"video_dataset/camera1/{current_Date}_VIDEO/" + f"{current_time}.avi",
                                                   fourcc,
                                                   20.0, (640, 480))
                video_writer.write(frame)
                # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            except:
                print("camera1 not found")

            try:
                ret2, frame2 = cam2.read()
                fgmask2 = fgbg.apply(frame2)
                try:
                    ret2, imBin2 = cv2.threshold(fgmask2, 200, 255, cv2.THRESH_BINARY)
                    mask2 = cv2.morphologyEx(imBin2, cv2.MORPH_OPEN, kernelOp)
                    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernelCl)
                except:
                    # print('EOF')
                    # print('UP:', cnt_up)
                    # print('DOWN:', cnt_down)
                    break

                contours0, hierarchy = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours0:
                    area = cv2.contourArea(cnt)
                    if area > areaTH:
                        M = cv2.moments(cnt)
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        x, y, w, h = cv2.boundingRect(cnt)
                        new = True
                        if cy in range(up_limit, down_limit):
                            for i in persons:
                                if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                                    # el objeto esta cerca de uno que ya se detecto antes
                                    new = False
                                    i.updateCoords(cx, cy)  # actualiza coordenadas en el objeto and resets age
                                    if i.going_UP(line_down, line_up) == True:
                                        if self.checkBox5.isChecked():
                                            playsound(beepsound)
                                        self.cnt_up += 1
                                    elif i.going_DOWN(line_down, line_up) == True:
                                        if self.checkBox4.isChecked():
                                            playsound(beepsound)
                                        self.cnt_down += 1
                                    break
                                if i.getState() == '1':
                                    if i.getDir() == 'down' and i.getY() > down_limit:
                                        i.setDone()
                                    elif i.getDir() == 'up' and i.getY() < up_limit:
                                        i.setDone()


                                if i.timedOut():
                                    # sacar i de la lista persons
                                    index = persons.index(i)
                                    persons.pop(index)
                                    del i  # liberar la memoria de i
                            if new == True:
                                p = MyPerson(pid, cx, cy, max_p_age)
                                persons.append(p)
                                pid += 1

                        # img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                str_up = 'UP: ' + str(self.cnt_up)
                str_down = 'DOWN: ' + str(self.cnt_down)
                frame2 = cv2.polylines(frame2, [pts_L1], False, line_down_color, thickness=2)
                frame2 = cv2.polylines(frame2, [pts_L2], False, line_up_color, thickness=2)
                frame2 = cv2.polylines(frame2, [pts_L3], False, (255, 255, 255), thickness=1)
                frame2 = cv2.polylines(frame2, [pts_L4], False, (255, 255, 255), thickness=1)
                cv2.putText(frame2, str_up, (10, 40), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame2, str_up, (10, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(frame2, str_down, (10, 90), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame2, str_down, (10, 90), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                self.textEdit6.setText(str(self.cnt_down))
                self.textEdit7.setText(str(self.cnt_up))
                self.displayImage2(frame2)

                # FOR SAVE AS VIDEO @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                if time.time() - start > duration2:
                    now = datetime.now()
                    # current_time = now.strftime("%H.%M.%S")

                    start = time.time()
                    if self.checkBox3.isChecked():
                        video_writer2 = cv2.VideoWriter(f"video_dataset/camera2/{current_Date}_VIDEO/" + f"{current_time}.avi",
                                                   fourcc,
                                                   20.0, (640, 480))
                video_writer2.write(frame2)
                # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

            except:
                print("camera2 not found")

            # cv2.imshow('Frame', frame2)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        cam2.release()
        cam.release()
        cv2.destroyAllWindows()
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")

    def displayImage(self, img, window=True):
        qformat = QtGui.QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QtGui.QImage.Format_RGBA8888
            else:
                qformat = QtGui.QImage.Format_RGB888
        outImage = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        if window:
            self.label_6.setPixmap(QtGui.QPixmap.fromImage(outImage))

    def displayImage2(self, img, window=True):
        qformat = QtGui.QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QtGui.QImage.Format_RGBA8888
            else:
                qformat = QtGui.QImage.Format_RGB888
        outImage = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        if window:
            self.label_9.setPixmap(QtGui.QPixmap.fromImage(outImage))

app = QApplication(sys.argv)
window = Window1()

window.show()
try:
    sys.exit(app.exec_())

except:
    print("exit")
