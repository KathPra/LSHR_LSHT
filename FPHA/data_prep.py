# Import packages
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Create functions
def create_skel(training_sample):
    """
    Requires: training_sample (saved as individual text files, 2D: Time x (Coordinate, Joint))

    Returns: 1 skeleton containing Sample x Coordinate x Time x Joint
    """
    x = np.loadtxt(training_sample)
    x = x[:,1:]
    xcor = np.empty([x.shape[0],0])
    ycor = np.empty([x.shape[0],0])
    zcor = np.empty([x.shape[0],0])

    for i in range(int(x.shape[1])):
        if i % 3 == 0:
            xcor = np.append(xcor, np.expand_dims(x[:,i], axis = 1), axis = 1)
            ycor = np.append(ycor, np.expand_dims(x[:,i+1], axis = 1), axis = 1)
            zcor = np.append(zcor, np.expand_dims(x[:,i+2], axis =1), axis = 1)

    skel = np.stack((xcor,ycor,zcor), axis = 0)

    return skel

def normalize_skel_length(skel):
    """
    Requires: skeleton data as numpy array (C x T x V)
    Returns: Skeleton of pre-defined length 64
    """
    window = 64
    if skel.shape[1]==64:
        skel_norm = skel
    else:
        C,T,V = skel.shape
        data = torch.tensor(skel,dtype=torch.float)
        data = data.permute(0, 2, 1).contiguous().view(C * V, T)
        data = data[None, None, :, :]
        data = F.interpolate(data, size=(C * V, window), mode='bilinear',align_corners=False).squeeze() # could perform both up sample and down sample
        skel_norm = data.contiguous().view(C, V, window).permute(0, 2, 1).contiguous().numpy()
    return skel_norm


# Retrieve all file names from the train set & test set
with open('train.txt') as f:
    train_samples = [line.rstrip('\n') for line in f]

with open('test.txt') as f:
    test_samples = [line.rstrip('\n') for line in f]


# Create labels for actions
train_labels = [line[:line.rfind("/")] for line in train_samples]
train_labels = [line[line.find("/")+1:] for line in train_labels]

test_labels = [line[:line.rfind("/")] for line in test_samples]
test_labels = [line[line.find("/")+1:] for line in test_labels]

# Create label dict
unique_label = list(set(train_labels))
unique_label.sort()
label_id = np.arange(0,45)
label_dict = {unique_label[i]: label_id[i] for i in range(len(unique_label))}

with open("action_labels.txt", 'w') as f: 
    for key, value in label_dict.items(): 
        f.write('%s : %s\n' % (key, value))
f.close()

# create label file for training
train_y = []
test_y = []

for action in train_labels:
    train_y.append(label_dict[action])

for action in test_labels:
    test_y.append(label_dict[action])


# Create train & test set (N, C, T, V, M) -> T varies, interpolation needed
train_set = np.empty([0, 3, 64,21])
skel_length_train = []
for file in train_samples:
    path = os.path.join(file[:file.rfind(" ")]+"/skeleton.txt")
    skel = create_skel(path)
    skel_length_train.append(skel.shape[1])
    skel_norm = normalize_skel_length(skel)
    train_set = np.append(train_set, np.expand_dims(skel_norm, axis=0), axis =0)
train_set = np.expand_dims(train_set, axis=4)
print("train set: ",train_set.shape)

# Create train set (N, C, T, V, M) -> T varies, interpolation needed
test_set = np.empty([0, 3, 64,21])
skel_length_test = []
for file in test_samples:
    path = os.path.join(file[:file.rfind(" ")]+"/skeleton.txt")
    skel = create_skel(path)
    skel_length_test.append(skel.shape[1])
    skel_norm = normalize_skel_length(skel)
    test_set = np.append(test_set, np.expand_dims(skel_norm, axis=0), axis =0)
test_set = np.expand_dims(test_set, axis=4)
print("test set: ",test_set.shape)

# save as npz file
np.savez("firstpersonhandaction.npz", x_train=train_set, y_train=train_y, x_test=test_set, y_test=test_y)


#print(skel_length)

## Plot Sample Length Distribution
"""
plt.hist(skel_length_train, range=[0, 150])
plt.savefig("vis/distribution_train.png")
plt.close()

plt.hist(skel_length_train, range=[0, 150])
plt.savefig("vis/distribution_test.png")
plt.close()

plt.hist([skel_length_train, skel_length_test], range=[0, 150], histtype='bar')
plt.savefig("vis/distribution_comp_S.png")
plt.close()

plt.hist([skel_length_train, skel_length_test], histtype='bar')
plt.savefig("vis/distribution_comp.png")
plt.close()
"""
