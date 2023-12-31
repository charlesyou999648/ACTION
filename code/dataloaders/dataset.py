import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from torchvision.transforms import RandomResizedCrop
import h5py
from scipy.ndimage import zoom
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
from torchvision.transforms import *
from PIL.ImageEnhance import *
from PIL import Image
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
from PIL import ImageFilter
import warnings
import math
from torch.distributions.beta import Beta
from torchvision.transforms import functional as F

# from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


def _compute_intersection(box1, box2):
    i1, j1, h1, w1 = box1
    i2, j2, h2, w2 = box2
    x_overlap = max(0, min(j1+w1, j2+w2) - max(j1, j2))
    y_overlap = max(0, min(i1+h1, i2+h2) - max(i1, i2))
    return x_overlap * y_overlap


class BaseDataSetsWithIndex(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None, index=16, label_type=0):
        self._base_dir = base_dir
        self.index = index
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train' and 'ACDC' in base_dir:
            with open(self._base_dir + '/train_slices.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
            if(label_type==1):
                self.sample_list = self.sample_list[:index]
            else:
                self.sample_list = self.sample_list[index:]
        elif self.split == 'train' and 'MM' in base_dir:
            with open(self._base_dir + '/train_slices.txt', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('.h5\n', '')
                                for item in self.sample_list]
            if(label_type==1):
                self.sample_list = self.sample_list[:index]
            else:
                self.sample_list = self.sample_list[index:]

        elif self.split == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num-index]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/data/slices/{}.h5".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.split == "train" and self.transform!=None:
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample
    
    
class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train' and 'ACDC' in base_dir:
            with open(self._base_dir + '/train_slices.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
        elif self.split == 'train' and 'MM' in base_dir:
            with open(self._base_dir + '/train_slices.txt', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('.h5\n', '')
                                for item in self.sample_list]

        elif self.split == 'val' and 'ACDC' in base_dir:
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
        elif self.split == 'val' and 'MM' in base_dir:
            with open(self._base_dir + '/test_vol.txt', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/data/slices/{}.h5".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.split == "train" and self.transform!=None:
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample
    
    
def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # if random.random() > 0.5:
        #     image, label = random_rot_flip(image, label)
        # elif random.random() > 0.5:
        #     image, label = random_rotate(image, label)
        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size
        # self.index=index
    def __call__(self, sample):
        # if self.index==0:
        image, label = sample['image'], sample['label']
        # else:
        #     image, label = sample[1]['image'], sample[1]['label']
        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph),], mode='constant', constant_values=0)

        (w, h) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]

        # if self.index==0:
            # return [{'image': image, 'label': label}, sample[1]]
        # else:
            # return [sample[0], {'image': image, 'label': label}]
        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size
        # self.index = index
    def __call__(self, sample):
        # if self.index==0:
        #     image, label = sample[0]['image'], sample[0]['label']
        # else:
        #     image, label = sample[1]['image'], sample[1]['label']

        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)

        (w, h) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]

        # if self.index==0:
        #     return [{'image': image, 'label': label}, sample[1]]
        # else:
        #     return [sample[0], {'image': image, 'label': label}]
        return {'image': image, 'label': label}


class RandomCropBatch(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size
        # self.index = index
    def __call__(self, sample):

        image, label = sample['image'], sample['label']
        new_image = []
        new_label = []
        # print(image.shape)
        for i in range(image.shape[0]):
            cur_image = image[i]
            cur_label = label[i]
        # pad the sample if necessary
            if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1]:
                pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
                ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
                cur_image = np.pad(cur_image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
                cur_label = np.pad(cur_label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
            # print(image[0].shape) # (180, 150, 88)
            # print(self.output_size[0]) # 112
            # exit()
            (w, h) = image[0].shape
            # print(w)
            # if np.random.uniform() > 0.33:
            #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
            #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
            # else:
            w1 = np.random.randint(0, w - self.output_size[0])
            h1 = np.random.randint(0, h - self.output_size[1])

            cur_label = cur_label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
            cur_image = cur_image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]

            new_image.append(cur_image)
            new_label.append(cur_label)

        # if self.index==0:
        #     return [{'image': image, 'label': label}, sample[1]]
        # else:
        #     return [sample[0], {'image': image, 'label': label}]
        new_image = torch.FloatTensor(np.array(new_image))
        new_label = torch.FloatTensor(np.array(new_label))
        return {'image': new_image, 'label': new_label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """
    # def __init__(self, index=0) -> None:
    #     self.index=index

    def __call__(self, sample):
        # if self.index==0:
        #     # print(np.shape(sample))
        #     image, label = sample[0]['image'], sample[0]['label']
        # else:
        #     image, label = sample[1]['image'], sample[1]['label']
        image, label = sample['image'], sample['label']
        # print(image.shape) # torch.Size([4, 180, 150, 88])
        # print(image.shape)
        for i in range(image.shape[0]):
            cur_img = image[i]
            cur_label = label[i]
            k = np.random.randint(0, 4)
            cur_img = np.rot90(cur_img, k)
            cur_label = np.rot90(cur_label, k)
            axis = np.random.randint(0, 2)
            cur_img = np.flip(cur_img, axis=axis).copy()
            cur_label = np.flip(cur_label, axis=axis).copy()

            image[i] = torch.FloatTensor(cur_img)
            label[i] = torch.FloatTensor(cur_label)
        # if self.index==0:
        #     return [{'image': image, 'label': label}, sample[1]]
        # else:
        #     return [sample[0], {'image': image, 'label': label}]
        # print('after rotation',image.shape)
        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, sample):
        if np.random.uniform(low=0, high=1, size=1) > self.p:
            return sample
        else: 
            image, label = sample['image'], sample['label']
            new_image = []
            sigma = random.uniform(0.15, 1.15)
            for i in range(image.shape[0]):
                image_i = ToPILImage()(image[i, 0, :, :]).filter(ImageFilter.GaussianBlur(radius=sigma))
                new_image.append(np.array(image_i)/255)

            image = torch.tensor(np.array(new_image), dtype=torch.float64)
            return {'image': image, 'label': label}


class RandomColorJitter(object):
    def __init__(self, color = (0.4, 0.4, 0.4, 0.1), p=0.1) -> None:
        self.color = color
        self.p = p
    
    def __call__(self, sample):
        if np.random.uniform(low=0, high=1, size=1) > self.p:
            return sample
        else:
            image, label = sample['image'], sample['label']
            for j in range(image.shape[0]):
                image[j, :, :, :] = ColorJitter(
                    brightness=self.color[0], 
                    contrast=self.color[1], 
                    saturation=self.color[2], 
                    hue=self.color[3])((image[j, :, :, :]))
                    

            return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, *image.shape).astype(np.float64)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample[0]['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


class Resize(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        pose = transforms.Compose([
            transforms.Resize((1, 256, 256))])
        image = pose(image)
        return {'image': image, 'label': label}


class BrightnessTransform(object):
    def __init__(self, p=0.5, mu=0.8, sigma=0.1) -> None:
        self.mu = mu
        self.sigma = sigma
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(low=0, high=1, size=1) > self.p:
            return sample
        else:
            image, label = sample['image'], sample['label']
            for j in range(image.shape[0]):
                image[j, :, :, :] = torch.clamp(self.mu*image[j, :, :, :] + self.sigma, min=0.0, max=1.0)
            return {'image': image, 'label': label}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable) # changes here
    # def infinite_shuffles():
    #     while True:
    #         yield np.random.permutation(iterable)
    # return itertools.chain.from_iterable(infinite_shuffles())


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


def worker_init_fn(worker_id):
    random.seed(100+worker_id)


class CustomMultiCropping(object):
    """ This class implements a custom multi-cropping strategy. In particular, 
    we generate the following crops:

    - N_large random crops of random size (default: 0.2 to 1.0) of the orginal size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. The crops
    are finally resized to the given size (default: 160). 

    - N_small random crops of random size (default: 0.05 to 0.14) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. The crops
    are finally resized to the given size (default: 96). There is the possibility to condition
    the smaller crops on the last large crop. Note that the latter is used as the key for MoCo.

    Args:
        size_large: expected output size for large crops
        scale_large: range of size of the origin size cropped for large crops
        
        size_small: expected output size for small crops
        scale_small: range of size of the origin size cropped for small crops

        N_large: number of large crops
        N_small: number of small crops
        
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR

        condition_small_crops_on_key: condition small crops on key
    """
    def __init__(self, size_large=160, scale_large=(0.2, 1.0), 
                    size_small=96, scale_small=(0.05, 0.14), N_large=2, N_small=4, 
                    ratio=(3. / 4., 4. / 3.), interpolation=F.InterpolationMode.BILINEAR,
                    condition_small_crops_on_key=True):
        if isinstance(size_large, (tuple, list)):
            self.size_large = size_large
        else:
            self.size_large = (size_large, size_large)
        
        if isinstance(size_small, (tuple, list)):
            self.size_small = size_small
        else:
            self.size_small = (size_small, size_small)

        if (scale_large[0] > scale_large[1]) or (scale_small[0] > scale_small[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation

        self.scale_large = scale_large
        self.scale_small = scale_small

        self.N_large = N_large
        self.N_small = N_small

        self.ratio = ratio
        self.condition_small_crops_on_key = condition_small_crops_on_key

    @staticmethod
    def get_params(img, scale, ratio, ):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def get_params_conditioned(self, img, scale, ratio, constraint):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
            constraint (tuple): params (i, j, h, w) that should be used to constrain the crop

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width
        for counter in range(10):
            rand_scale = random.uniform(*scale)
            target_area = rand_scale * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                intersection = _compute_intersection((i, j, h, w), constraint)
                if intersection >= 0.1 * target_area: # 10 percent of the small crop is part of big crop.
                    return i, j, h, w
        
        return self.get_params(img, scale, ratio) # Fallback to default option

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            multi_crop (list of lists): result of multi-crop
        """
        multi_crop = []
        multi_crop_params = []
        for ii in range(self.N_large):
            i, j, h, w = self.get_params(img, self.scale_large, self.ratio)
            multi_crop_params.append((i, j, h, w))
            multi_crop.append(F.resized_crop(img, i, j, h, w, self.size_large, self.interpolation))

        for ii in range(self.N_small):
            if not self.condition_small_crops_on_key:
                i, j, h, w = self.get_params(img, self.scale_small, self.ratio)

            else:
                i, j, h, w = self.get_params_conditioned(img, self.scale_small, self.ratio,
                                                            multi_crop_params[self.N_large -1])
                
            multi_crop_params.append((i, j, h, w))
            multi_crop.append(F.resized_crop(img, i, j, h, w, self.size_small, self.interpolation))

        return multi_crop, multi_crop_params 

    def __repr__(self):
        format_string = self.__class__.__name__ + '(size_large={0}'.format(self.size_large)
        format_string += ', scale_large={0}'.format(tuple(round(s, 4) for s in self.scale_large))
        format_string += ', size_small={0}'.format(tuple(round(s, 4) for s in self.size_small))
        format_string += ', scale_small={0}'.format(tuple(round(s, 4) for s in self.scale_small))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', condition_small_crops_on_key={})'.format(self.condition_small_crops_on_key)
        return format_string



if __name__ == '__main__':
    
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    root_path = '/data/data/MM-WHS'
    patch_size_larger = (256, 256)
    patch_size = (224, 224)

    # transform_student = transforms.Compose([
    #                         # RandomCropBatch(patch_size),
    #                         RandomRotFlip(),
    #                         # RandomColorJitter(p=0.8, color = (0.2, 0.2, 0.2, 0.1)),
    #                         # RandomNoise(mu=0.1, sigma=2.0),
    #                     ])

    # transform_teacher = transforms.Compose([
    #                         # RandomCropBatch(patch_size),
    #                         RandomRotFlip(),
    #                     ])
    db_train = BaseDataSets(base_dir=root_path, split="train", transform=transforms.Compose([
                        RandomCrop(patch_size_larger),
                        ]))
    
    all_index = len(db_train)
    print(all_index)
    # print(db_train)
    labeled_idxs = list(range(16))
    
    unlabeled_idxs = list(range(16, all_index))
    batch_size = 4
    labeled_bs = 2
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    for i_batch, batch in enumerate(trainloader):
        # print(batch['image'].shape)
        # teacher_batch = transform_teacher(batch)
        # student_batch = transform_student(batch)
        teacher_batch, teacher_label = batch['image'], batch['label']
        student_batch, student_label = batch['image'], batch['label']

        # student_batch = student_batch.unsqueeze(1)
        # teacher_batch = teacher_batch.unsqueeze(1)
        print(torch.max(student_label[1]))
        # print(student_batch.shape)
        # print(student_label.shape)

        # print(torch.mean(teacher_batch))
        # print(torch.mean(student_batch))
        # isEqual = teacher_label.eq(student_label)
        # if not (isEqual.all()):
        #     print('problem at index ', i_batch)
        #     exit()

# print('no problem')