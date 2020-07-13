# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import glob
import cv2
import torch
import numpy as np
import time
import scipy.io as sio
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm
import argparse
from PIL import Image
import face_detection
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader

#l2_norm
def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output
#Get the face in the center
def get_final_box(w,h,boxes):
    temp=[]
    for box in boxes:
        l=pow((box[0]+box[2]-w)/2,2)+pow((box[1]+box[3]-h)/2,2)
#         l=abs((box[0]+box[2]-w)/2)+abs((box[1]+box[3]-h)/2,2)
        temp.append(l)
    nums=np.argsort(temp)
    final_box=boxes[nums[0]]
#     print(final_box[-1])
    return final_box[:4]

#A square box is obtained by taking the detected face box as the center
def change(box):
    centure=[int((box[0]+box[2])/2),int((box[1]+box[3])/2)]
    if box[2]-box[0]>box[3]-box[1]:
        bias=int(((box[2]-box[0])-(box[3]-box[1]))/2)
        box[3]=box[3]+bias
        box[1]=box[1]-bias
    else:
        bias=int(((box[3]-box[1])-(box[2]-box[0]))/2)
        box[2]=box[2]+bias
        box[0]=box[0]-bias        
    return box
#Expand the area of the face box
def change_2(box):
    centure=[int((box[0]+box[2])/2),int((box[1]+box[3])/2)]
    w=box[2]-box[0]
    h=box[3]-box[1]
    return [int(centure[0]-0.6*w),int(centure[1]-0.6*h),int(centure[0]+0.6*w),int(centure[1]+0.6*h)]

#get all image name in one folder
def get_name(path):
    paths = []
    for i in os.listdir(path):
        paths.append(i)
    return paths

#get a name-feature dict
def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        temp=features[i]
        fe_dict[each] = temp
    return fe_dict


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

#load the image and inference to get the face vector
def inference_dataload(model, face_path, tta=True):

    def default_loader(path):
        return Image.open(path).convert('RGB')

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    if tta:
        p=1.0
    else:
        p=0.0
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=p),
        transforms.Resize((112, 112),interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std), ])

    class MyDataset(Dataset):
        def __init__(self, path, transform=transform, loader=default_loader):
            paths = []
            for i in os.listdir(path):
                paths.append(path + '/' + i)
            self.paths = paths
            self.transform = transform
            self.loader = loader

        def __getitem__(self, index):
            img_path = self.paths[index]
            img = self.loader(img_path)
            if self.transform is not None:
                img = self.transform(img)
            return img

        def __len__(self):
            return len(self.paths)
    model.eval()
    with torch.no_grad():
        test_data = MyDataset(path=face_path)
        test_loader = DataLoader(dataset=test_data, batch_size=128, num_workers=0)
        temp = []
        for step, imgs in enumerate(test_loader):
            output = model(imgs.cuda())
            f = output.data
            temp.append(f)
        source_embs = torch.cat(temp)
        source_embs = source_embs.cpu()
        source_embs=l2_norm(source_embs).numpy()
    return source_embs

if __name__ == '__main__':
    model_path1='model/model_1.pth'
    model_path2='model/model_2.pth'
    model_path3='model/model_3.pth'
    model_path4='model/model_4.pth'
    model_path5='model/model_5.pth'
    model_path6='model/model_6.pth'
    model_path7='model/model_7.pth'
    model_path8='model/model_8.pth'


    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('test_path',
                        default='./test/',
                        type=str)
    parser.add_argument('output_predictions_path',
                    default='./predictions/',
                    type=str)
    args = parser.parse_args()

    old_path=args.test_path
    predictions_path=args.output_predictions_path
    new_path='./new_test/'
    feature_path='./feature/'
    score_path='./score/'

    if not os.path.exists(old_path):
        print('the orign test image path is not Existing')
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    if not os.path.exists(score_path):
        os.makedirs(score_path)
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)            
#===========================================================
# get the croped face by use dsfd module 
    imgs=os.listdir(old_path)
    temps=[int(img.split('.')[0]) for img in imgs]
    temps=sorted(temps)
    imgs=[old_path+str(temp)+'.jpg' for temp in temps]

    # file='bad_ones.txt'
    detector = face_detection.build_detector(
        "DSFDDetector",
        confidence_threshold=.7,
        nms_iou_threshold=.3,
        max_resolution=1080
    )

    t0 = time.time()
    for img_name in imgs:
        img_cv2 = cv2.imread(img_name)
        img=Image.open(img_name)
        w,h=img.size
        dets = detector.detect(img_cv2[:, :, ::-1])[:, :4]
        if len(dets) == 0:
            final_box=[int(w*7/24),int(h*7/24),int(w*17/24),int(h*17/24)]
            final_box=change(final_box)
            final_box=change_2(final_box)
            img_warped=img.crop(final_box)
            img_warped = img_warped.resize((112, 112),Image.BICUBIC)
            img_warped.save(img_name.replace(old_path,new_path))
            # print(img_name)
            # with open(file,'a+') as f:
            #     f.write(img_name)
        else:
            final_box=get_final_box(w,h,dets)
            final_box=change(final_box)
            final_box=change_2(final_box)
            img_warped=img.crop(final_box)
            img_warped = img_warped.resize((112, 112),Image.BICUBIC)
            img_warped.save(img_name.replace(old_path,new_path))
    t1 = time.time()
    print(t1-t0)
#===========================================================
#get the face feature
    test_path=new_path
    name_list=get_name(test_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_list=[model_path1,model_path2,model_path3,model_path4,model_path5,model_path6,model_path7,model_path8]
    models_list1=[model_path1,model_path4,model_path5,model_path6,model_path7]
    models_list2=[model_path2,model_path3,model_path8]
    model = IR_152([112, 112]).to(device)
    for model_path in models_list1:
        name=model_path.split('/')[1].split('.')[0]

        model.load_state_dict(torch.load(model_path))
        features=inference_dataload(model,test_path, tta=True)
        fe_dict = get_feature_dict(name_list, features)
        print('Output number:', len(fe_dict))
        sio.savemat(feature_path+name+'_filp.mat', fe_dict)

        features=inference_dataload(model,test_path, tta=False)
        fe_dict = get_feature_dict(name_list, features)
        print('Output number:', len(fe_dict))
        sio.savemat(feature_path+name+'_nofilp.mat', fe_dict)


    model = IR_50([112, 112]).to(device)
    for model_path in models_list2:
        name=model_path.split('/')[1].split('.')[0]

        model.load_state_dict(torch.load(model_path))
        features=inference_dataload(model,test_path, tta=True)
        fe_dict = get_feature_dict(name_list, features)
        print('Output number:', len(fe_dict))
        sio.savemat(feature_path+name+'_filp.mat', fe_dict)

        features=inference_dataload(model,test_path, tta=False)
        fe_dict = get_feature_dict(name_list, features)
        print('Output number:', len(fe_dict))
        sio.savemat(feature_path+name+'_nofilp.mat', fe_dict)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#get the face feature Similarity
    for model_name in models_list:
        name=model_name.split('/')[1].split('.')[0]

        face_features1 = sio.loadmat(feature_path+name+'_filp.mat')
        face_features2 = sio.loadmat(feature_path+name+'_nofilp.mat')
        print('Loaded mat')

        sample_sub = open('random_predictions.csv', 'r')
        sub = open(score_path+name+'.csv', 'w')
        print('Loaded CSV')
        lines = sample_sub.readlines()[1:]
        sub.write('TEMPLATE_ID1'+','+'TEMPLATE_ID2'+','+'SCORE'+'\n')
        print(len(lines))
        pbar = tqdm(total=len(lines))
        for line in lines:
            a = line.split(',')[0]
            b = line.split(',')[1]
            sub.write(a+','+b+',')
            features1=face_features1[a+'.jpg'][0]+face_features2[a+'.jpg'][0]
            features2=face_features1[b+'.jpg'][0]+face_features2[b+'.jpg'][0]

            features1=features1/np.linalg.norm(features1, 2)
            features2=features2/np.linalg.norm(features2, 2)
            
            score = cosin_metric(features1, features2)

            temp='%.5f'%score
            sub.write(temp+'\n')
            pbar.update(1)

        sample_sub.close()
        sub.close()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#combine base models to get  an assembled model    
    sample_sub1 = open(score_path+'model_1.csv', 'r')
    sample_sub2 = open(score_path+'model_2.csv', 'r')
    sample_sub3 = open(score_path+'model_3.csv', 'r')
    sample_sub4 = open(score_path+'model_4.csv', 'r')
    sample_sub5 = open(score_path+'model_5.csv', 'r')
    sample_sub6 = open(score_path+'model_6.csv', 'r')
    sample_sub7 = open(score_path+'model_7.csv', 'r')
    sample_sub8 = open(score_path+'model_8.csv', 'r')
    sub= open(predictions_path+'predictions.csv', 'w')
    print('Loaded CSV')
    lines1 = sample_sub1.readlines()[1:]
    lines2 = sample_sub2.readlines()[1:]
    lines3 = sample_sub3.readlines()[1:]
    lines4 = sample_sub4.readlines()[1:]
    lines5 = sample_sub5.readlines()[1:]
    lines6 = sample_sub6.readlines()[1:]
    lines7 = sample_sub7.readlines()[1:]
    lines8 = sample_sub8.readlines()[1:]
    sub.write('TEMPLATE_ID1'+','+'TEMPLATE_ID2'+','+'SCORE'+'\n')
    print(len(lines1))
    pbar = tqdm(total=len(lines1))
    for i in range(len(lines1)):
        a= lines1[i].split(',')[0]
        b= lines1[i].split(',')[1]
        sub.write(a+','+b+',')
        score1=float(lines1[i].split(',')[-1])
        score2=float(lines2[i].split(',')[-1])
        score3=float(lines3[i].split(',')[-1])
        score4=float(lines4[i].split(',')[-1])
        score5=float(lines5[i].split(',')[-1])
        score6=float(lines6[i].split(',')[-1])
        score7=float(lines7[i].split(',')[-1])
        score8=float(lines8[i].split(',')[-1])
        score=(score1+score2+score3+score4+0.4*score5+0.4*score6+0.4*score7+0.4*score8)/5.6
        temp='%.5f'%score
        sub.write(temp+'\n')
        pbar.update(1)
    sample_sub1.close()
    sample_sub2.close()
    sample_sub3.close()
    sample_sub4.close()
    sample_sub5.close()
    sample_sub6.close()
    sample_sub7.close()
    sample_sub8.close()        
    sub.close()