# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'SPS_tool.ui'
##
## Created by: Qt User Interface Compiler version 5.14.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt5.QtCore import (QCoreApplication, QMetaObject, QObject, QPoint,
    QRect, QSize, QUrl, Qt)
# from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
#     QFontDatabase, QIcon, QLinearGradient, QPalette, QPainter, QPixmap,
#     QRadialGradient)
# from PySide2.QtWidgets import *

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import *

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(900, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setObjectName(u"centralwidget")
        self.imageArea = QLabel(self.centralwidget)
        self.imageArea.setObjectName(u"imageArea")
        self.imageArea.setGeometry(QRect(40, 50, 640, 360))
        self.imageArea.setFrameShape(QFrame.Box)
        self.imageArea.setMidLineWidth(1)
        self.loadVideo = QPushButton(self.centralwidget)
        self.loadVideo.setObjectName(u"loadVideo")
        self.loadVideo.setGeometry(QRect(750, 50, 120, 50))
        self.loadImage = QPushButton(self.centralwidget)
        self.loadImage.setObjectName(u"loadImage")
        self.loadImage.setGeometry(QRect(750, 140, 120, 50))
        self.gateArea1 = QPushButton(self.centralwidget)
        self.gateArea1.setObjectName(u"gateArea1")
        self.gateArea1.setGeometry(QRect(750, 230, 120, 51))
        self.save = QPushButton(self.centralwidget)
        self.save.setObjectName(u"save")
        self.save.setGeometry(QRect(750, 680, 120, 50))
        self.travelArea = QPushButton(self.centralwidget)
        self.travelArea.setObjectName(u"travelArea")
        self.travelArea.setGeometry(QRect(750, 410, 120, 50))
        self.parkingArea = QPushButton(self.centralwidget)
        self.parkingArea.setObjectName(u"parkingArea")
        self.parkingArea.setGeometry(QRect(750, 500, 120, 50))
        self.notification = QLabel(self.centralwidget)
        self.notification.setObjectName(u"notification")
        self.notification.setGeometry(QRect(240, 700, 401, 31))
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.notification.setFont(font)
        self.ok = QPushButton(self.centralwidget)
        self.ok.setObjectName(u"ok")
        self.ok.setGeometry(QRect(750, 590, 120, 50))
        self.gateArea2 = QPushButton(self.centralwidget)
        self.gateArea2.setObjectName(u"gateArea2")
        self.gateArea2.setGeometry(QRect(750, 320, 120, 51))
        # self.background = QLabel(self.centralwidget)
        # self.background.setObjectName(u"background")
        # self.background.setGeometry(QRect(0, 0, 891, 741))
        # self.background.setAutoFillBackground(False)
        # self.background.setFrameShape(QFrame.Box)


        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 900, 26))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)


        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Smart Parking System", None))
        self.imageArea.setText("")
        self.loadVideo.setText(QCoreApplication.translate("MainWindow", u"Load video", None))
        self.loadImage.setText(QCoreApplication.translate("MainWindow", u"Load image", None))
        self.gateArea1.setText(QCoreApplication.translate("MainWindow", u"Gate Area 1", None))
        self.save.setText(QCoreApplication.translate("MainWindow", u"Save", None))
        self.travelArea.setText(QCoreApplication.translate("MainWindow", u"Travel area", None))
        self.parkingArea.setText(QCoreApplication.translate("MainWindow", u"Parking area", None))
        self.notification.setText("")
        self.ok.setText(QCoreApplication.translate("MainWindow", u"OK", None))
        self.gateArea2.setText(QCoreApplication.translate("MainWindow", u"Gate Area 2", None))
        #self.background.setText("")
    # retranslateUi