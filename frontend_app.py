# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MMTraffic.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
import os
import tkinter as tk
from tkinter import filedialog
from shutil import copyfile
#from test2 import run


class Ui_MainWindow(QtWidgets.QWidget):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1346, 867)
        font = QtGui.QFont()
        font.setPointSize(10)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 10, 361, 331))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(20, 80, 55, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(20, 120, 55, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(20, 40, 55, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.comboBox = QtWidgets.QComboBox(self.groupBox)
        self.comboBox.setGeometry(QtCore.QRect(100, 40, 111, 22))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.comboBox.setFont(font)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox_2 = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_2.setGeometry(QtCore.QRect(100, 80, 111, 22))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_3 = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_3.setGeometry(QtCore.QRect(100, 120, 111, 22))
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(20, 160, 71, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.comboBox_4 = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_4.setGeometry(QtCore.QRect(100, 160, 111, 22))
        self.comboBox_4.setObjectName("comboBox_4")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setGeometry(QtCore.QRect(20, 200, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.comboBox_5 = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_5.setGeometry(QtCore.QRect(100, 200, 111, 22))
        self.comboBox_5.setObjectName("comboBox_5")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.timeEdit = QtWidgets.QTimeEdit(self.groupBox)
        self.timeEdit.setGeometry(QtCore.QRect(100, 240, 91, 22))
        self.timeEdit.setObjectName("timeEdit")
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setGeometry(QtCore.QRect(20, 240, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.groupBox)
        self.label_8.setGeometry(QtCore.QRect(210, 240, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.timeEdit_2 = QtWidgets.QTimeEdit(self.groupBox)
        self.timeEdit_2.setGeometry(QtCore.QRect(250, 240, 91, 22))
        self.timeEdit_2.setObjectName("timeEdit_2")
        self.label_9 = QtWidgets.QLabel(self.groupBox)
        self.label_9.setGeometry(QtCore.QRect(20, 280, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.dateEdit = QtWidgets.QDateEdit(self.groupBox)
        self.dateEdit.setGeometry(QtCore.QRect(100, 280, 91, 22))
        self.dateEdit.setObjectName("dateEdit")
        self.label_10 = QtWidgets.QLabel(self.groupBox)
        self.label_10.setGeometry(QtCore.QRect(210, 280, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.dateEdit_2 = QtWidgets.QDateEdit(self.groupBox)
        self.dateEdit_2.setGeometry(QtCore.QRect(250, 280, 91, 22))
        self.dateEdit_2.setObjectName("dateEdit_2")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 360, 241, 211))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.textEdit = QtWidgets.QTextEdit(self.groupBox_2)
        self.textEdit.setGeometry(QtCore.QRect(10, 120, 221, 81))
        self.textEdit.setObjectName("textEdit")
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_5.setGeometry(QtCore.QRect(20, 20, 201, 28))
        self.pushButton_5.setObjectName("pushButton_5")
        self.toolButton = QtWidgets.QToolButton(self.groupBox_2)
        self.toolButton.setGeometry(QtCore.QRect(20, 60, 71, 41))
        font = QtGui.QFont()
        font.setPointSize(22)
        self.toolButton.setFont(font)
        self.toolButton.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.toolButton.setObjectName("toolButton")
        self.toolButton_2 = QtWidgets.QToolButton(self.groupBox_2)
        self.toolButton_2.setGeometry(QtCore.QRect(150, 60, 71, 41))
        font = QtGui.QFont()
        font.setPointSize(22)
        self.toolButton_2.setFont(font)
        self.toolButton_2.setObjectName("toolButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(10, 690, 93, 28))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(160, 690, 93, 28))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(400, 10, 381, 521))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.graphicsView = QtWidgets.QGraphicsView(self.groupBox_3)
        self.graphicsView.setGeometry(QtCore.QRect(20, 30, 341, 441))
        self.graphicsView.setObjectName("graphicsView")
        self.pushButton = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton.setGeometry(QtCore.QRect(30, 480, 93, 28))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_2.setGeometry(QtCore.QRect(260, 480, 93, 28))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(10, 580, 241, 101))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_4.setFont(font)
        self.groupBox_4.setObjectName("groupBox_4")
        self.textEdit_2 = QtWidgets.QTextEdit(self.groupBox_4)
        self.textEdit_2.setGeometry(QtCore.QRect(10, 30, 221, 61))
        self.textEdit_2.setObjectName("textEdit_2")
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(810, 10, 521, 351))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_5.setFont(font)
        self.groupBox_5.setObjectName("groupBox_5")
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.groupBox_5)
        self.graphicsView_2.setGeometry(QtCore.QRect(20, 30, 481, 231))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.label_6 = QtWidgets.QLabel(self.groupBox_5)
        self.label_6.setGeometry(QtCore.QRect(180, 280, 161, 31))
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.pushButton_6 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_6.setGeometry(QtCore.QRect(20, 310, 93, 28))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_7 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_7.setGeometry(QtCore.QRect(410, 310, 93, 28))
        self.pushButton_7.setObjectName("pushButton_7")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox_5)
        self.lineEdit.setGeometry(QtCore.QRect(120, 310, 281, 22))
        self.lineEdit.setObjectName("lineEdit")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1346, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton_6.clicked.connect(self.openFile)
        self.pushButton_3.clicked.connect(self.Search)
        #self.pushButton_7.clicked.connect(self.process)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MM-Traffic-Event-Tracking"))
        self.groupBox.setTitle(_translate("MainWindow", "Search Option"))
        self.label_2.setText(_translate("MainWindow", "Object"))
        self.label_3.setText(_translate("MainWindow", "Action"))
        self.label.setText(_translate("MainWindow", "Class"))
        self.comboBox.setItemText(0, _translate("MainWindow", "None"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Crash"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Animals"))
        self.comboBox.setItemText(3, _translate("MainWindow", "Collapse"))
        self.comboBox.setItemText(4, _translate("MainWindow", "Fire"))
        self.comboBox.setItemText(5, _translate("MainWindow", "Flooding"))
        self.comboBox.setItemText(6, _translate("MainWindow", "Landslide"))
        self.comboBox.setItemText(7, _translate("MainWindow", "Snow"))
        self.comboBox.setItemText(8, _translate("MainWindow", "Treefall"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "None"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "Car"))
        self.comboBox_2.setItemText(2, _translate("MainWindow", "Human"))
        self.comboBox_2.setItemText(3, _translate("MainWindow", "Bike"))
        self.comboBox_2.setItemText(4, _translate("MainWindow", "Moto Bike"))
        self.comboBox_2.setItemText(5, _translate("MainWindow", "Animals"))
        self.comboBox_3.setItemText(0, _translate("MainWindow", "None"))
        self.comboBox_3.setItemText(1, _translate("MainWindow", "Turn left "))
        self.comboBox_3.setItemText(2, _translate("MainWindow", "Turn right"))
        self.comboBox_3.setItemText(3, _translate("MainWindow", "Turn around"))
        self.comboBox_3.setItemText(4, _translate("MainWindow", "Go straight"))
        self.label_4.setText(_translate("MainWindow", "Position"))
        self.comboBox_4.setItemText(0, _translate("MainWindow", "None"))
        self.comboBox_4.setItemText(1, _translate("MainWindow", "In front of"))
        self.comboBox_4.setItemText(2, _translate("MainWindow", "Behind"))
        self.comboBox_4.setItemText(3, _translate("MainWindow", "Left"))
        self.comboBox_4.setItemText(4, _translate("MainWindow", "Right"))
        self.label_5.setText(_translate("MainWindow", "Location"))
        self.comboBox_5.setItemText(0, _translate("MainWindow", "None"))
        self.comboBox_5.setItemText(1, _translate("MainWindow", "On street"))
        self.label_7.setText(_translate("MainWindow", "Time"))
        self.label_8.setText(_translate("MainWindow", "To"))
        self.label_9.setText(_translate("MainWindow", "Date"))
        self.label_10.setText(_translate("MainWindow", "To"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Topology"))
        self.pushButton_5.setText(_translate("MainWindow", "Detail Class-based search"))
        self.toolButton.setText(_translate("MainWindow", "+"))
        self.toolButton_2.setText(_translate("MainWindow", "-"))
        self.pushButton_3.setText(_translate("MainWindow", "Search"))
        self.pushButton_4.setText(_translate("MainWindow", "Refine"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Result (Key Frame)"))
        self.pushButton.setText(_translate("MainWindow", "Refresh"))
        self.pushButton_2.setText(_translate("MainWindow", "Export"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Query (free style)"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Image Classification"))
        self.label_6.setText(_translate("MainWindow", "Type"))
        self.pushButton_6.setText(_translate("MainWindow", "Open file"))
        self.pushButton_7.setText(_translate("MainWindow", "Process"))

    def Search(self):
        topology = self.comboBox.currentText() + ' ;' + self.comboBox_2.currentText() + ' ;' + self.comboBox_3.currentText() + ' ;' + self.comboBox_4.currentText() + ' ;' + self.comboBox_5.currentText()
        self.textEdit.setText(topology)
        print(topology)

    def checkPath(self, image_path):
        dst = r'D:/2_COMPUTER SCIENCE/MASTER DATASCIENCE K30/Thesis/code/road-incidents-thesis-master/incidents_cleaned/test_temp/Upload/'
        image_path = image_path 
        nametag = os.path.splitext(os.path.basename(image_path))[0] + os.path.splitext(os.path.basename(image_path))[1]
        print(nametag)
        copyfile(image_path, dst + nametag)
        self.lineEdit.setText(image_path)
        if os.path.isfile(image_path):
            scene = QtWidgets.QGraphicsScene(self)
            pixmap = QPixmap(image_path)
            item = QtWidgets.QGraphicsPixmapItem(pixmap)
            scene.addItem(item)
            self.graphicsView_2.setScene(scene)
            

#     def process(self):
#         image_path = self.lineEdit.text()
#         dst = r'D:/2_COMPUTER SCIENCE/MASTER DATASCIENCE K30/Thesis/code/road-incidents-thesis-master/incidents_cleaned/test_temp/Upload/'
#         nametag = os.path.splitext(os.path.basename(image_path))[0] + os.path.splitext(os.path.basename(image_path))[1]
#         type_name = run()
#         self.label_6.setText(type_name)
#         os.remove(dst + nametag)

    def openFile(self):
        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename()
        self.checkPath(file_path)
        print(file_path)
    



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
