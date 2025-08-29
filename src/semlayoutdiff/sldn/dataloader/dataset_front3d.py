import torch
import numpy as np
from torch.utils.data import DataLoader
# from torchflow.data.loaders.nde.image import MNIST
from torchvision.transforms import RandomHorizontalFlip, Pad, RandomAffine, \
    CenterCrop, RandomCrop, Compose, ToPILImage, ToTensor
import math

from .front3d.front3d_fast import Front3DFast
# from .voxelroom.voxelroom_fast import VoxelRoomFast


def add_data_args(parser):
    # Data params
    parser.add_argument('--dataset', type=str, default='front3d', )

    # Train params
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_categories', type=int, default=22)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', type=eval, default=False)
    parser.add_argument('--augmentation', type=str, default=None)
    parser.add_argument('--floor_plan', type=eval, default=False)
    parser.add_argument('--wo_floor', type=eval, default=False)
    parser.add_argument('--data_size', type=int, default=120)
    parser.add_argument('--data_dir', type=str, default='livingroom')
    parser.add_argument('--room_type_condition', type=eval, default=False)
    parser.add_argument('--w_arch', type=eval, default=False)
    parser.add_argument('--specific_room_type', type=str, default=None)
    parser.add_argument('--text_condition', type=eval, default=False)
    parser.add_argument('--mix_condition', type=eval, default=False)

def get_data_id(args):
    return '{}'.format(args.dataset)


def get_augmentation(augmentation, dataset, data_shape):
    h, w = data_shape
    if augmentation is None:
        pil_transforms = []
    elif augmentation == 'horizontal_flip':
        pil_transforms = [RandomHorizontalFlip(p=0.5)]
    elif augmentation == 'shift':
        pad_h, pad_w = int(0.07 * h), int(0.07 * w)
        if 'cityscapes' in dataset and 'large' in dataset:
            # Annoying, cityscapes images have a 3-border around every image.
            # This messes up shift augmentation and needs to be dealt with.
            assert h == 128 and w == 256
            print('Special cityscapes transform')
            pad_h, pad_w = int(0.075 * h), int(0.075 * w)
            pil_transforms = [CenterCrop((h - 2, w - 2)),
                              RandomHorizontalFlip(p=0.5),
                              Pad((pad_h, pad_w), padding_mode='edge'),
                              RandomCrop((h - 2, w - 2)),
                              Pad((1, 1), padding_mode='constant', fill=3)]

        else:
            pil_transforms = [RandomHorizontalFlip(p=0.5),
                              Pad((pad_h, pad_w), padding_mode='edge'),
                              RandomCrop((h, w))]
    elif augmentation == 'neta':
        assert h == w
        pil_transforms = [Pad(int(math.ceil(h * 0.04)), padding_mode='edge'),
                          RandomAffine(degrees=0, translate=(0.04, 0.04)),
                          CenterCrop(h)]
    elif augmentation == 'eta':
        assert h == w
        pil_transforms = [RandomHorizontalFlip(),
                          Pad(int(math.ceil(h * 0.04)), padding_mode='edge'),
                          RandomAffine(degrees=0, translate=(0.04, 0.04)),
                          CenterCrop(h)]

    # torchvision.transforms.s
    return pil_transforms


def get_augmentation_3d(augmentation, dataset, data_shape):
    h, w, d = data_shape
    if augmentation is None:
        pil_transforms = []

    # torchvision.transforms.s
    return pil_transforms


def get_data(args):
    if args.dataset == 'front3d':
        data_shape = (1, args.data_size, args.data_size)
        num_classes = args.num_categories
        if args.wo_floor:
            num_classes -= 1
        pil_transforms = get_augmentation(args.augmentation, args.dataset,
                                          (args.data_size, args.data_size))
        pil_transforms = Compose(pil_transforms)
        if not hasattr(args, 'data_dir'):
            # old checkpoint has different args structure
            args.data_dir = args.room_type
            args.specific_room_type = None
        train = Front3DFast(root="datasets", split=args.data_dir,
                            resolution=(args.data_size, args.data_size),
                            transform=pil_transforms, floor_plan=args.floor_plan, wo_floor=args.wo_floor,
                            room_type_condition=args.room_type_condition, w_arch=args.w_arch, specific_room_type=args.specific_room_type, 
                            text_condition=args.text_condition, mixed_condition=args.mix_condition)
        
        if args.room_type_condition:
            # create a weighted random sampler
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=train.weights,
                num_samples=len(train),
                replacement=True
            )

            # Data Loader
            train_loader = DataLoader(train, batch_size=args.batch_size, num_workers=args.num_workers,
                                    sampler=sampler, pin_memory=args.pin_memory)
        else:
            train_loader = DataLoader(train, batch_size=args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, pin_memory=args.pin_memory)
        eval_loader = DataLoader(train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                 pin_memory=args.pin_memory)

        return train_loader, eval_loader, data_shape, num_classes
