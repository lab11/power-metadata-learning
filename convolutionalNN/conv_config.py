#data and labels
model_save_path = "./dev9/model.ckpt"
train_data = "../data/dev9/train.npy"
train_labels = "../data/dev9/trainLabels.npy"
train_ids = "../data/dev9/trainID.npy"
test_data = "../data/dev9/unseen.npy"
test_labels = "../data/dev9/unseenLabels.npy"
test_ids = "../data/dev9/unseenID.npy"

test = True

#add_pf
use_pf = True

#the size of the first max pooling layer
pre_pool_size = 10
pre_pool_stride = 10

#conv1 layer
conv1_filter_size = 100
conv1_num_filters = 32
pool1_size = 10
pool1_stride = 10

#conv2 layer
conv2_filter_size = 100
conv2_num_filters = 64
pool2_size = 4
pool2_stride = 4

#hidden layer
hidden1_size = 1024

#dropout
keep_prob = 0.5

#learning rate
lr = 1e-4
