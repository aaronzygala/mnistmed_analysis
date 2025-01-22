class Config:

    # Model configurations
    LR = 0.001
    GAMMA=0.1

    # Default arguments for main program
    DEFAULT_DATA_FLAG = "pathmnist"
    DEFAULT_OUTPUT_ROOT = "./output"
    DEFAULT_NUM_EPOCHS = 5
    DEFAULT_IMAGE_SIZE = 28
    DEFAULT_GPU_IDS = '0'
    DEFAULT_BATCH_SIZE = 128
    DEFAULT_MODEL_FLAG = 'resnet18'
    DEFAULT_MODEL_NAME = 'model1'