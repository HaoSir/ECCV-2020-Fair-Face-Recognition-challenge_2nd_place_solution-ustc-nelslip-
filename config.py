import torch


configurations = {
    1: dict(
        SEED = 1337, # random seed for reproduce results

        DATA_ROOT = './data', # the parent root where your train/val/test data are stored
        MODEL_ROOT = './work_space/save', # the root to buffer your checkpoints
        LOG_ROOT = './work_space/log', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = './work_space/models/Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = None, # the root to resume training from a saved checkpoint

        val_pair_path=None,
        val_data_path=None,

        train_name='train',
        BACKBONE_NAME = 'IR_152', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 256,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.01, # initial LR
        # NUM_EPOCH = 50, # total epoch number (use the firt 1/25 epochs to warm up)
        # WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        # MOMENTUM = 0.9,
        # STAGES = [22,37,46], # epoch stages to decay learning rate
        NUM_EPOCH = 60, # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [25,48,54], # epo

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = True, # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID = [0,1,2,3], # specify your GPU ids
        PIN_MEMORY = True,
        NUM_WORKERS = 6,
),
}
