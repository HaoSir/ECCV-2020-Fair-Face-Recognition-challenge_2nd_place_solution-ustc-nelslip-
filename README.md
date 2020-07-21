
# ECCV 2020 Fair Face Recognition challenge 2nd_place_solution
## reference resources

1.[insightface](https://github.com/deepinsight/insightface)

2.[face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)


## Requirements
as shown in requirements.txt
## Instructions to reproduce the test prediction
### for docker users:
1. you need have installed docker and nvidia-docker.
2. you should download the docker file from the [google drive](https://drive.google.com/file/d/1MxBHarqjvWNsl8UAS5mf0oJQetRWBQaT/view?usp=sharing)
3. use the command to load the  docker file :docker load -i ustc_nelslip_submitv1.zip
4. prepare your test images and random predictions files, besides, you should prepare a "predictions" folder to contain the generated final submitted prediction file.These three parameters are necessary.
5. run the code to get your final submitted prediction file in the folder "predictions" as with the following command:

sudo nvidia-docker run  -ti  -v  $test_image_path:/test/  -v $random_predictions_path:/random_predictions.csv -v  $output_predictions_path:/predictions/ ustc_nelslip_submit:v1.0   ./test/ ./predictions/

1. $test_image_path -- any arbitrary directory contains input test images，like /home/xxx/ECCV/ECCV_docker/test/
2. $random_predictions_path -- any arbitrary file path to the random prediction file ,like /home/xxx/ECCV/ECCV_docker/random_predictions.csv
3. $output_predictions_path -- any arbitrary directory, and the  final submitted prediction file would be generated in the directory. like /home/xxx/ECCV/ECCV_docker/predictions/

## Instructions to retrain our models
1. prepare the orign train data, and the pretrained model from :
[DSFD model](https://drive.google.com/file/d/1lX8-Nqcsx9qGw-ok563wqPJ_cpPkIOiU/view?usp=sharing), 
[IR_152 model](https://drive.google.com/file/d/1g41T38fanW857kA9oTowi5dndVLibGAV/view?usp=sharing), 
[IR_50 model](https://drive.google.com/file/d/1LO1KRu8DFoHfBQjFt8nR1QIl5fOnBEXR/view?usp=sharing)
2. use the command to obtain the face from given image: python crop_face.py --old_path xx --new_path xx
3. change your config with config.py, and train your model with the command: python train_demo.py


**we choose two backbone: IR_50, IR_152, two head: arcface, cosface and multiple data enhancement methods, and the detailed information would be shown in a paper soon.**
