import sys
# pip install pyqt5
import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from gui_fr import Ui_MainWindow
import os

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import *
from shapely.geometry import Point, Polygon

list_poly=[]
list_poly_parking=[]
list_gates=[]

num_polies=0
num_polies_parking=0
num_gates=0

eps_x=40
eps_y=50

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        self.uic.loadVideo.clicked.connect(self.loadVideo)
        self.uic.loadImage.clicked.connect(self.loadImage)
        self.uic.gateArea1.clicked.connect(self.gateArea1)
        self.uic.gateArea2.clicked.connect(self.gateArea2)
        self.uic.travelArea.clicked.connect(self.travelArea)
        self.uic.parkingArea.clicked.connect(self.parkingArea)
        self.uic.save.clicked.connect(self.save)
        self.uic.ok.clicked.connect(self.ok)

        # img_background=cv2.imread('./astronaut.jpg')
        # img_background=cv2.resize(img_background, (891,741))
        # qt_img = self.convert_cv_qt(img_background)
        # self.uic.background.setPixmap(qt_img)

        self.label = self.uic.imageArea
        self.image=None
        self.lastPoint=None
        self.x=None
        self.y=None

        self.isTravelSelection=None
        self.isParkingSelection=None
        self.isGateSelection=None

        self.isWhichSelection = -1 # 0: Travel; 1: Parking; 2: Gate1 ; 3: Gate2 ; -1: chua chon chuc nang them vung

        self.travelArea_ele=[]
        self.parkingArea_ele=[]
        self.gateArea1_ele=[]
        self.gateArea2_ele=[]
        self.gateArea_ele=[]  #gom 2 phan tu area1 va area2z

        self.rand_int=None


    def ok(self):
        if self.isWhichSelection==0:
            list_poly.append(self.travelArea_ele)
            self.uic.notification.setText("Đã thêm một vùng di chuyển")
        elif self.isWhichSelection==1:
            list_poly_parking.append(self.parkingArea_ele)
            self.uic.notification.setText("Đã thêm một vùng đỗ xe")
        elif self.isWhichSelection==2:
            self.gateArea_ele.append(self.gateArea1_ele)
            self.uic.notification.setText("Đã thêm vị trí 1 của cổng")
        elif self.isWhichSelection==3:
            self.gateArea_ele.append(self.gateArea2_ele)
            if len(self.gateArea_ele)==2:
                list_gates.append(self.gateArea_ele)
            else:
                self.uic.notification.setText("Chưa chọn vùng 1 của cổng")
                return
            self.uic.notification.setText("Đã thêm vị trí 2 của cổng")
    
    def changeVideoSize(self,path):
        cap = cv2.VideoCapture(path)


        # Get video metadata
        video_fps = cap.get(cv2.CAP_PROP_FPS),

        # we are using x264 codec for mp4
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        writer = cv2.VideoWriter('../OUTPUT.mp4', apiPreference=0, fourcc=fourcc,
                            fps=video_fps[0], frameSize=(640,360))

        while True:
            ret, frame = cap.read()
            if not ret: break # break if cannot receive frame
            # convert to grayscale
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame=cv2.resize(frame,(640,360))
            
            writer.write(frame) # write frame
                
        # release and destroy windows
        writer.release()
        cap.release()

    def loadVideo(self):
        f = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*);;Python File (*.py)", )
        path=f[0]

        cap=cv2.VideoCapture(path)

        d=0
        image=None
        isChangeSize=False
        while True:
            d +=1
            stt, fr=cap.read()
            if d==3:
                image=fr
                if image.shape[0]!=360 or image.shape[1]!=640:
                    isChangeSize=True
                break
            if not stt:
                break
        image=cv2.resize(image, (640,360))
        self.image=image
        self.displayImg(image)
        if isChangeSize:
            self.changeVideoSize(path)



    def loadImage(self):
        f = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*);;Python File (*.py)", )
        path=f[0]

        image=cv2.imread(path)
        image=cv2.resize(image, (640,360))
        self.image=image
        self.displayImg(image)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()

    def gateArea1(self):
        self.rand_int =np.random.randint(100, 255)
        self.gateArea_ele=[] #dua gateArea_ele ve trang thai rong
        self.gateArea1_ele=[]
        self.isWhichSelection=2
        global num_gates
        num_gates +=1
        self.uic.notification.setText("Chọn vùng 1 của cổng "+str(num_gates))

    def gateArea2(self):
        self.rand_int =np.random.randint(100, 255)
        self.gateArea2_ele=[]
        self.isWhichSelection=3
        self.uic.notification.setText("Chọn vùng 2 của cổng "+str(num_gates))

    def travelArea(self):
        global num_polies
        num_polies +=1
        self.rand_int =np.random.randint(100, 255)
        self.travelArea_ele=[]
        self.uic.notification.setText("Chọn vùng di chuyển thứ "+str(num_polies))
        self.isWhichSelection=0

    def parkingArea(self):
        global num_polies_parking 
        num_polies_parking +=1
        self.rand_int =np.random.randint(100, 255)
        self.parkingArea_ele=[]
        self.isWhichSelection=1
        self.uic.notification.setText("Chọn vùng đỗ xe thứ "+str(num_polies_parking))

    def save(self):
        if self.isWhichSelection==0:
            f=open("./travel_area.txt", "w")
            f.write(str(list_poly))
            f.close()
            self.uic.notification.setText("Đã lưu vùng di chuyển")
            self.isWhichSelection=-1
        elif self.isWhichSelection==1:
            f=open("./parking_area.txt", "w")
            f.write(str(list_poly_parking))
            f.close()
            self.uic.notification.setText("Đã lưu vùng đỗ xe")
            self.isWhichSelection=-1
        elif self.isWhichSelection==2:
            self.uic.notification.setText("Phải chọn đủ 2 vị trí của 1 cổng")
        elif self.isWhichSelection==3 and len(self.gateArea_ele)==2:
            f=open("./gate_area.txt", "w")
            f.write(str(list_gates))
            f.close()
            self.uic.notification.setText("Đã thêm cổng")
            self.isWhichSelection=-1
        elif self.isWhichSelection==-1:
            self.uic.notification.setText("Chọn trước năng thêm vùng tương ứng trước")
        else:
            self.uic.notification.setText("Thêm vùng 1 của cổng trước khi thêm vùng 2")

    def displayImg(self, img):
    	qt_img = self.convert_cv_qt(img)
    	self.uic.imageArea.setPixmap(qt_img)

    
    def saveImg(self):
    	cv2.imwrite(os.path.join(des, self.img_arr[self.idx]), self.image)
    	self.idx +=1
    	img=cv2.imread(os.path.join(path, self.img_arr[self.idx]))
    	self.image=img
    	self.displayImg(img)


    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 360, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


    def checkValidSelection(self,x,y):
        if x<40:
            return False
        if x>680:
            return False
        if y<50:
            return False
        if y>410:
            return False
        return True

    def mousePressEvent(self, event):
        self.uic.notification.setText("")
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()
            self.x, self.y =self.lastPoint.x(), self.lastPoint.y()
            if not self.checkValidSelection(self.x,self.y):
                self.uic.notification.setText("Chỉ chọn tọa độ trong vùng ảnh")
            else:
                if self.isWhichSelection==0:
                    self.travelArea_ele.append((self.x-eps_x, self.y-eps_y))
                    self.image =cv2.circle(self.image, (self.x-40, self.y-50), radius=1 , color=(self.rand_int, 0, self.rand_int), thickness=2)
                    self.displayImg(self.image)
                elif self.isWhichSelection==1:
                    self.parkingArea_ele.append((self.x-eps_x, self.y-eps_y))
                    self.image =cv2.circle(self.image, (self.x-40, self.y-50), radius=1 , color=(0, self.rand_int, self.rand_int), thickness=2)
                    self.displayImg(self.image)
                elif self.isWhichSelection==2:
                    self.gateArea1_ele.append((self.x-eps_x, self.y-eps_y))
                    self.image =cv2.circle(self.image, (self.x-40, self.y-50), radius=1 ,color=(self.rand_int, 0, 0), thickness=2)
                    self.displayImg(self.image)
                elif self.isWhichSelection==3:
                    self.gateArea2_ele.append((self.x-eps_x, self.y-eps_y))
                    self.image =cv2.circle(self.image, (self.x-40, self.y-50), radius=1 ,color=(0, self.rand_int, 0), thickness=2)
                    self.displayImg(self.image)




    def mouseReleaseEvent(self, event):
        pass




if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())