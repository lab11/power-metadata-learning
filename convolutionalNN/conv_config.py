#data and labels
model_save_path = "./checkpoint1/model.ckpt"
train_data = "../data/sets/train.npy"
train_labels = "../data/sets/trainLabels.npy"
train_ids = "../data/sets/trainID.npy"
test_data = "../data/sets/unseen.npy"
test_labels = "../data/sets/unseenLabels.npy"
test_ids = "../data/sets/unseenID.npy"

test = False

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
