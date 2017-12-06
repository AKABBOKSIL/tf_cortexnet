import os

import jy_makeList as list
import jy_tf_Main2

#리스트 관련
listOnOff = False #이미지 리스트 뽑는 여부 결정

root = 'D:/myFolder/'
folderList = ['1', '2', '3', '4'] # real name of folders name in root

trainTxt = root+'imgTrainingList.txt' # txt file of training image directory
testTxt = root+'imgTestList.txt'# txt file of test image directory

tr_cnt=1000 # of images for training each class
test_cnt =200 # of images for test each class
saver_name='datasetName' #input the dataset name

##################################
#모델 관련
_model_ = 'resnet34' #model name
filter = 64 #initial # of filteres
picSize = 128 # size of image
imgType= 'jpg' #'png' #type of image

batch_size = 32
test_batchSize = 10
epoch = 256


#####################################
############auto setting#############

channel = 3
num_classes = len(folderList)

total_image = num_classes*tr_cnt
iteration = int(epoch * total_image/batch_size)
total_test_cnt = num_classes * test_cnt

#save log
save_accuracy = root + 'accuracy.txt'
save_valid = root + 'validation.txt'
save_saver = root + '0.ckpt/' +_model_+'/'
save_confusion = root + 'confusion_matrix.txt'
save_fp = root + 'fp.txt'


if __name__ == '__main__':
    if listOnOff:
        list.makelist_main()
    if not os.path.exists(save_saver):
        os.mkdir(save_saver)

    jy_tf_Main2.main()

