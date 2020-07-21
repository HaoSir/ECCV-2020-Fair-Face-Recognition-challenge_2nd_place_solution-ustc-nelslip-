import glob
import os
import cv2
import time
import face_detection
import numpy as np
import os
from tqdm import tqdm
import argparse
from PIL import Image
import numpy as np
parser = argparse.ArgumentParser(
    description='face Detector Training With Pytorch')
parser.add_argument('--old_path',
                    default='test_image',
                    type=str)
parser.add_argument('--new_path',
                    default='save_image',
                    type=str)
args = parser.parse_args()
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
def change_2(box):
    centure=[int((box[0]+box[2])/2),int((box[1]+box[3])/2)]
    w=box[2]-box[0]
    h=box[3]-box[1]
    return [int(centure[0]-0.6*w),int(centure[1]-0.6*h),int(centure[0]+0.6*w),int(centure[1]+0.6*h)]
old_path=args.old_path
new_path=args.new_path
if not os.path.exists(old_path):
    print('the orign test image path is not Existing')
if not os.path.exists(new_path):
    os.makedirs(new_path);

imgs=os.listdir(old_path)
temps=[int(img.split('.')[0]) for img in imgs]
temps=sorted(temps)
imgs=[old_path+'/'+str(temp)+'.jpg' for temp in temps]

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
        