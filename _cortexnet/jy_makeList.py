#! /usr/bin/env_python
# -*-coding:utf-8-*-
"""
    make list for training or test from directory
    output: ...list.txt

"""
import random, os
import glob
import jy_tf_param as param


def make_list_glob(path, label):
    roots = param.root
    # training
    cnt = 0
    if(param.treeOnOff):
        files = glob.glob(roots + path + '/aug/*')
    else:
        files = glob.glob(roots + path + '/*')
    trainingList = []

    for file in files:
        if cnt < param.tr_cnt:
            name = '%s\t' % file + '%d\n' % label
            trainingList.append(name)
            cnt += 1

    # test
    files.sort(reverse=True)
    testList = []
    cnt = 0
    for file in files:
        if cnt < param.test_cnt:
            name = '%s\t' % file + '%d\n' % label
            testList.append(name)
            cnt += 1

    return trainingList, testList


def makelist_main():

    imgTrainingList = []
    imgTestList = []

    for c in range(param.num_classes):
        list1, list2 = make_list_glob(param.folderList[c], c)
        imgTrainingList += list1
        imgTestList += list2

    random.shuffle(imgTrainingList)
    file_name = param.trainTxt
    f = open(file_name, 'w')

    # write list on notepad
    for line in imgTrainingList:
        f.write(line)
    f.close()
    print('training')

    random.shuffle(imgTestList)
    file_name = param.testTxt
    f = open(file_name, 'w')
    for line in imgTestList:
        f.write(line)
    f.close()
    print('test')

def makelist_main2():
    """
    train/test 폴더가 분리되어 있을 경우
    한번에 리스트 만들고 txt파일 저장하는 것까지 함
    """

    imgTrainingList = []
    imgTestList = []
    cnt = 0

    for c in range(param.num_classes):
        roots = param.root
        # training

        trainingList = []
        files = glob.glob(roots + param.folderList[c] + '/train/*')
        for file in files:
            if cnt<param.tr_cnt:
                name = '%s\t' % file + '%d\n' % c
                trainingList.append(name)
            cnt+=1

        # test
        testList = []
        files = glob.glob(roots + param.folderList[c] + '/test/*')
        cnt=0

        for file in files:
            if cnt<param.test_cnt:
                name = '%s\t' % file + '%d\n' % c
                testList.append(name)
            cnt+= 1
        imgTrainingList += trainingList
        imgTestList += testList

    random.shuffle(imgTrainingList)
    file_name = param.trainTxt
    f = open(file_name, 'w')

    # write list on notepad
    for line in imgTrainingList:
        f.write(line)
    f.close()
    print('training')

    file_name = param.testTxt
    f = open(file_name, 'w')
    for line in imgTestList:
        f.write(line)
    f.close()
    print('test')