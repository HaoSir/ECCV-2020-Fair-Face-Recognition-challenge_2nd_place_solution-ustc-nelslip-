introduce of the folder and the file:
Data Processing——face_detection：we use DSFD face detection algorithm to get cropped faces.

Face recognition——backbone：we defined many different backbones, such as ResNet_152,IR_SE_152
	         ——head：we defined many different head metrics,such as ArcFace, CosFace, SphereFace
	         ——loss：we defined  a focal loss to get better classification effect
	         ——util：we defined  many functions to make our code simple and effective，just like warm_up function and function of balancing train data.
	         ——work_space：use for training stage.

data folder——test：the folder should contain original test images.
	——new_test：the folder would contain generated cropped test faces.
	——feature：the folder would contain generated face features.
	——score：the folder would contain many generated predictions files.
	——predictions：the folder would contain generated final submitted prediction file.

model folder——model:the folder contain eight pretrain models.

config.py——defined some basic parameters and variables.
train_demo.py——A train demo
random_predictions.csv——the random predictions file to generated final submitted prediction file, so it is necessary.
requirements.txt——the requirement packages
inference.py——we use this file to crop test faces and use the pretrain models to get final submitted prediction file.
run.sh——to run inference.py more simply.
#===============================================================================
Instruction for use the code locally to verification:

a) Installation packages in the requirements.txt.
b) before you run the code, you should download pretrained model from the link:https://drive.google.com/file/d/1lX8-Nqcsx9qGw-ok563wqPJ_cpPkIOiU/view?usp=sharingfor face detect 
and link:_ for Face recognition.
c) put the original test images in the folder "test",and rename your random_prediction file as random_predictions.csv
d) run the inference.py to get your final submitted prediction file in the folder "predictions" as with the following command:
python inference.py ./test/ ./predictions/

for train:
a) Installation packages in the requirements.txt.
b) you should download pretrained model from the link:https://drive.google.com/file/d/1lX8-Nqcsx9qGw-ok563wqPJ_cpPkIOiU/view?usp=sharing for face detect
 and link:_ for Face recognition.
c) put the croped train images in the folder "data/train/",and then change your settings for backbone or heads.
d)run the train_demo.py to train your models.

#==================================================

PS: Our detailed solution would be released gradually after the deadline  of  associated Workshop.
