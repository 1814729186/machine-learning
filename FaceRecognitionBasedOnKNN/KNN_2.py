import os
import cv2
import shutil
import math
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import dlib
import face_reco_from_camera as fa
# 1. Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()
# 2. Dlib 人脸 landmark 特征点检测器
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
# 3. Dlib Resnet 人脸识别模型，提取 128D 的特征矢量
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
#显示涉及到的人的图片
def show_people(flag = False):
    if flag:
        string = 'orl_faces\\s'
        rootPath = os.listdir('orl_faces')
        imgs = []
        # 7 * 6
        fileLen = len(rootPath)
        for i in range(1, fileLen):
            filepath = os.path.join(string + str(i), '1.pgm')
            imgs.append(mpimg.imread(filepath))
        plt.figure()
        count = 0
        for i in  range(1, 8):
            for j in range(1, 7):
                plt.subplot(6, 7, count + 1)
                plt.imshow(imgs[count])
                if count < 39:
                    count += 1
        plt.show()

# 拷贝别人的特征提取的类
class Hog_descriptor():
    def __init__(self, img, cell_size=16, bin_size=8):
        self.img = img
        self.img = np.sqrt(img / np.max(img))
        self.img = img * 255
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 / self.bin_size

    #特征提取方法
    def extract(self):
        height, width = self.img.shape
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)

        cell_gradient_vector = np.zeros((int(height / self.cell_size), int(width / self.cell_size), self.bin_size))
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)
        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)
        return hog_vector, hog_image

    def global_gradient(self):
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        return idx, (idx + 1) % self.bin_size, mod

    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image

# 数据分类，训练数据、测试数据
def moveFile(srcFilePath, dstPath):
    if not os.path.isfile(srcFilePath):
        print("%s not exist!" %(srcFilePath))
    else:
        if not os.path.exists(dstPath):
            os.makedirs(dstPath)
        shutil.copy(srcFilePath, dstPath)


# 转化灰度图片为jpg
def bmpToJpg(filePath):
    if not os.path.isdir(filePath):
        raise ValueError("filePath Error!")
    for each in os.listdir(filePath):
        filename = each.split('.')[0] + ".jpg"
        im = Image.open(os.path.join(filePath, each))
        im.save(os.path.join(filePath, filename))


# 数据划分，将数据随机分类成训练数据和测试数据，参数propotion为数据划分比例（作为训练数据的图片百分比）
def makeSet(proportion):
    path = ['trainingSet', 'testingSet']
    for each in path:
        if not os.path.exists(each):
            os.mkdir(each)      #创建数据文件夹
    src = 'orl_faces\\s'

    for i in range(40):
        random_list = os.listdir(src + str(i + 1))      #文件目录表
        print("processing the number s{0} directory.....".format(i + 1))
        random.shuffle(random_list)
        k = proportion * 10
        for j in range(len(random_list)):   #遍历文件目录
            dstPath = path[0] if j < k else path[1]     #划分到不同数据文件夹
            filename = random_list[j]
            srcFilePath = os.path.join(src + str(i + 1), filename)
            moveFile(srcFilePath, dstPath)  #移动文件
            srcfile = os.path.join(dstPath, filename)
            dstfile = os.path.join(dstPath, str(10 * (i + 1) + int(filename.split('.')[0])) + '.pgm')
            os.rename(srcfile, dstfile)
            print("move samples from directory {0} ----> {1}".format(srcFilePath, dstfile))

# change the gray picture to row vector
def img2vector(filename):
    img = mpimg.imread(filename)

    return img.reshape(1, -1)

# 读取训练集
def readData(trainFilePath, testFilePath):
    trainingFileList = os.listdir(trainFilePath)
    testFileList = os.listdir(testFilePath)

    trainingLen = len(trainingFileList)
    trainLabels = []
    trainSet = np.zeros((trainingLen, 92 * 112)) # 92 * 112
    for i in np.arange(trainingLen):
        filename = trainingFileList[i]
        classNum = int(filename.split('.')[0])
        classNum = math.ceil(classNum / 10) - 1
        trainLabels.append(classNum)
        trainSet[i] = img2vector(os.path.join(trainFilePath, filename))

    testingLen = len(testFileList)
    testSet = np.zeros((testingLen, 92 * 112))
    testLabels = []
    for i in np.arange(testingLen):
        filename = testFileList[i]
        classNum = int(filename.split('.')[0])
        classNum = math.ceil(classNum / 10) - 1
        testLabels.append(classNum)
        testSet[i] = img2vector(os.path.join(testFilePath, filename))

    return {'trainSet' : trainSet, 'trainLabels' : trainLabels,
            'testSet' : testSet, 'testLabels' : testLabels}

# 进行归一化处理 normalization
def maxmin_norm(array):
    maxcols = array.max(axis = 0)
    mincols = array.min(axis = 0)
    data_shape = array.shape
    data_rows, data_cols = data_shape
    t = np.empty((data_rows, data_cols))
    for i in range(data_cols):
        t[:, i] = (array[:, i] - mincols[i]) / (maxcols[i] - mincols[i])
    return t

    
# KNN classifier
def kNNClassify(inX, dataSet, labels, k = 3,method = 'Euclidean'):
    if method == 'Euclidean':
        distance = np.sum(np.power((dataSet - inX), 2), axis = 1)  # 计算欧氏距离（平方）
    else:
        dis = dataSet - inX
        for i in range(len(dis)):
            for j in range(len(dis[i])):
                dis[i][j] = math.fabs(dis[i][j])                           #取绝对值
        distance = np.sum(np.power(dis, 1), axis = 1)   
        distance = distance*distance                                        #曼哈顿距离（平方）
    sortedArray = np.argsort(distance, kind = "quicksort")[:k]  #argsort函数返回的是数组值从小到大的索引值
    w = []
    for i in range(k):
        w.append((distance[sortedArray[k-1]] - distance[sortedArray[i]])\
                 / (distance[sortedArray[k-1]] - distance[sortedArray[0]]))
    count = np.zeros(41)
    temp = 0
    for each in sortedArray:
        count[labels[each]] += 1 + w[temp]
        temp += 1
    label = np.argmax(count) # 如果label中有多个一样的样本数，那么取第一个最大的样本类别
    return label

def main(k):
    '''
    #删除原有的分类数据
    if os.path.exists(r'./testingSet/'):
        shutil.rmtree(r'./testingSet/')
        print("delete testingSet")
    if os.path.exists(r'./trainingSet/'):
        shutil.rmtree(r'./trainingSet/')
        print("delete trainingSet")
    '''
    dist_Sce = input("1 欧氏距离 2 曼哈顿距离\n")
    if dist_Sce == '1':
        dist_method = 'Euclidean'
    else:
        dist_method = 'Manhattan'
    show_people(flag = False)    #显示40个人的数据信息
    # 数据划分
    if not os.path.exists('testingSet') and not os.path.exists("trainingSet"):
        prompt = "Please input the proportion(0~1): "
        while True:
            receive = input(prompt)
            proportion = round(float(receive), 1)
            if  0 < proportion < 1:
                break
            prompt = "Please input again ! :) :"
        makeSet(proportion)

    # 数据读入
    data = readData('trainingSet', 'testingSet')
    trainSet = data['trainSet']
    trainLabels = data['trainLabels']
    testSet = data['testSet']
    testLabels = data['testLabels']
    
    
    # 归一化处理
    temp = trainSet.shape[0]
    array = np.vstack((trainSet, testSet))
    normalized_array = maxmin_norm(array)
    trainSet, testSet = normalized_array[:temp], normalized_array[temp:]

    correct_count = 0
    test_number = testSet.shape[0]
    string = "test sample serial number: {0}, sample label: {1}, classify label: {2}------>correct?: {3}"
    for i in np.arange(test_number):
       label =  kNNClassify(testSet[i], trainSet, trainLabels, k = k,method = dist_method)
       if label == testLabels[i]:
           correct_count += 1
       print(string.format(i + 1, testLabels[i], label, label == testLabels[i]))

    print("face recognization accuracy: {}%".format((correct_count / test_number) * 100))
    return (correct_count / test_number) * 100

# verify the proper k
def selectK():
    x = list()
    y = list()
    for i in range(3, 11):
        x.append(i)
        y.append(main(i))
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    strSec = input("1 实时人脸识别\n2 实验数据人脸识别\n")
    if strSec == '1':
        Face_Recognizer_con = fa.Face_Recognizer()
        Face_Recognizer_con.run()
    else:
        while True:
            k_str = input("Please input k:\n")
            k = eval(k_str)
            if k > 0 and k < 8:
                main(k)
            else:
                break
