from dataset import *

dataset = LiTS_stage2(
        root='/media/adminm/New Volume/DLFS/Unet/sample',   
        lowerbound = 45,
        upperbound = 80,
        train=True,
        dev=False,
        transform=None,
        target_transform=None)

for i in range(len(dataset)):
    masked_image, masked_target, liver_mask, target = dataset[i]
    print(masked_image.shape, masked_target.shape, liver_mask.shape, target.shape)
    if i == 10:
        break