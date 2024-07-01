PATH ="./dataset/cats_and_dogs_small/"
SIZE = (48,48)

TRAIN_PATH = PATH + "train/"
TEST_PATH = PATH + "test/"
TRAIN_FNAME = PATH + "train-data.npz"
TEST_FNAME = PATH + "test-data.npz"

FILTERS = 64
KERNEL_SIZE = (3,3)
INPUT_SHAPE = (SIZE[0],SIZE[1],3)
UNITS = 32
NUM_OUTPUT = 2
BATCH_SIZE = 16
EPOCHS = 10
FNAME_MODEL = "./models/model1.bin"