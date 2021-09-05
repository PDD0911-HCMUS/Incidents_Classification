from frontend_app_new import *
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.QtGui import QPixmap
import os
import sys
import random
import cv2
import glob
import numpy as np
class My_Application(QDialog):
    def __init__(self):
        super().__init__()
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)
        self.image_path = []
        self.ui.pushButton_3.clicked.connect(self.checkPath)
        self.ui.pushButton_2.clicked.connect(self.make_frame)
        
        self.data_dir = 'incidents_cleaned'

    def checkPath(self):
        #image_path =  self.ui.comboBox.currentText()
        image_paths = self.Search()
        #scene1
        scene1 = QtWidgets.QGraphicsScene(self)
        pixmap = QPixmap(image_paths[0])
        item = QtWidgets.QGraphicsPixmapItem(pixmap)
        scene1.addItem(item)
        self.ui.graphicsView.setScene(scene1)
        #scene2
        scene2 = QtWidgets.QGraphicsScene(self)
        pixmap = QPixmap(image_paths[1])
        item = QtWidgets.QGraphicsPixmapItem(pixmap)
        scene2.addItem(item)
        self.ui.graphicsView_2.setScene(scene2)
        #scene3
        scene3 = QtWidgets.QGraphicsScene(self)
        pixmap = QPixmap(image_paths[2])
        item = QtWidgets.QGraphicsPixmapItem(pixmap)
        scene3.addItem(item)
        self.ui.graphicsView_3.setScene(scene3)
        #scene4
        scene4 = QtWidgets.QGraphicsScene(self)
        pixmap = QPixmap(image_paths[3])
        item = QtWidgets.QGraphicsPixmapItem(pixmap)
        scene4.addItem(item)
        self.ui.graphicsView_4.setScene(scene4)
        #scene5
        scene5 = QtWidgets.QGraphicsScene(self)
        pixmap = QPixmap(image_paths[4])
        item = QtWidgets.QGraphicsPixmapItem(pixmap)
        scene5.addItem(item)
        self.ui.graphicsView_5.setScene(scene5)
        #scene6
        scene6 = QtWidgets.QGraphicsScene(self)
        pixmap = QPixmap(image_paths[5])
        item = QtWidgets.QGraphicsPixmapItem(pixmap)
        scene6.addItem(item)
        self.ui.graphicsView_6.setScene(scene6)

        self.image_path = image_paths

        return self.image_path

    def Search(self):
        topology = self.ui.comboBox.currentText() + ' ;' + self.ui.comboBox_2.currentText() + ' ;' + self.ui.comboBox_3.currentText() + ' ;' + self.ui.comboBox_4.currentText() + ' ;' + self.ui.comboBox_5.currentText()
        self.ui.textEdit.setText(topology)
        class_image = self.ui.comboBox.currentText()
        image_paths = self.get_images(class_image)
        image_path_choice = random.choices(image_paths, k=6)
        #print(type(image_path_choice))
        return image_path_choice

    def get_images(self, root_filepath, sort=True):
        data_path = os.path.join(self.data_dir, 'train')
        image_paths = []
        for folder, _, imgs in os.walk(data_path):
            if(folder == os.path.join(data_path, root_filepath)):
                for image_path in imgs:
                    image_paths.append(os.path.join(folder, image_path))
        if sort is True:
            image_paths = sorted(image_paths)
        return image_paths
    
    def make_frame(self):
        print(self.image_path)
        img_array = []
        image_path = self.image_path
        for im in image_path:
            img = cv2.imread(im)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
        video_out = cv2.VideoWriter('class_result.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 150, size)
        for i in range(len(img_array)):
            video_out.write(img_array[i])
        video_out.release()
        return img_array
        
    # def make_video_result(self):
    #     img_array = self.make_frame
    #     video_out = cv2.VideoWriter('class_result.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    #     for i in range(len(img_array)):
    #         video_out.write(img_array[i])
    #     video_out.release()



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    class_instance = My_Application()
    class_instance.MainWindow.show()
    class_instance.ui.pushButton_3.clicked.connect(class_instance.checkPath)
    #class_instance.ui.pushButton_2.clicked(class_instance.checkPath)
    sys.exit(app.exec_())