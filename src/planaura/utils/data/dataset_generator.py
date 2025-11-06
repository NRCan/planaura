from planaura.utils.data.data_folder_parser_generator import fetch_parser
from planaura.utils.raster_management.read_geotiff import read_geotiff
import torch
import random
import cv2
import copy
import numpy as np
from torch.utils.data import Dataset
from timm.models.layers import to_2tuple


class CSVDataset(Dataset):

    def __init__(self, mode, data_file, config, transforms, sampler):

        self.transforms = transforms
        self.sampler = sampler
        paths_included_in_csvs = False if "paths_included_in_csvs" not in config else config["paths_included_in_csvs"]
        dataset_parser = fetch_parser(config, mode)
        dataset_parser.read_data(data_file, paths_included_in_csvs)
        self.database = dataset_parser.data_base

        self.database_index_list = sampler.return_list_index(self.database)

        img_size = to_2tuple(config['model_params']['img_size'])
        self.model_input_width = img_size[1]
        self.model_input_height = img_size[0]

        self.num_frames = config['num_frames']
        self.bands = config['model_params']['bands']

        self.use_xarray = False if 'use_xarray' not in config else config['use_xarray']

    def __len__(self):
        return len(self.database_index_list)

    def get_index_list(self):
        return self.database_index_list

    def __getitem__(self, index):
        imgs_input, imgs_target = self.read_image(index)
        if imgs_input[0] is not None:
            data = self.database[index]
            sample = copy.deepcopy(data)
            for fr in range(self.num_frames):
                sample['img_target_' + str(fr)] = imgs_target[fr]
                sample['img_input_' + str(fr)] = imgs_input[fr]
            if self.transforms:
                sample = self.transforms(sample)
            return sample
        else:
            random_index = random.randint(0, len(self.database_index_list) - 1)
            return self[self.database_index_list[random_index]]

    def polish_image_size(self, img, img_type):
        if img.shape[1] > self.model_input_width or \
                img.shape[0] > self.model_input_height:
            w = img.shape[1]
            h = img.shape[0]
            if w > h:
                new_h = int(self.model_input_width * h / w)
                new_w = self.model_input_width
                if new_h > self.model_input_height:
                    new_w = int(self.model_input_height * new_w / new_h)
                    new_h = self.model_input_height
                new_dim = (new_w, new_h)
            else:
                new_w = int(self.model_input_height * w / h)
                new_h = self.model_input_height
                if new_w > self.model_input_width:
                    new_h = int(self.model_input_width * new_h / new_w)
                    new_w = self.model_input_width
                new_dim = (new_w, new_h)

        if img_type == 'image':
            if img.shape[1] > self.model_input_width or \
                    img.shape[0] > self.model_input_height:
                copy_img = copy.deepcopy(img)
                img = np.zeros((new_h, new_w, copy_img.shape[2])).astype(copy_img.dtype)
                for ch in range(img.shape[2]):
                    img[:, :, ch] = cv2.resize(copy_img[:, :, ch], new_dim, interpolation=cv2.INTER_AREA)
            return img
        elif img_type == 'mask':
            if img.shape[1] > self.model_input_width or \
                    img.shape[0] > self.model_input_height:
                copy_img_mask = copy.deepcopy(img)
                img = cv2.resize(copy_img_mask, new_dim, interpolation=cv2.INTER_NEAREST_EXACT)
            return img
        else:
            return None

    def read_image(self, index):
        load_success_input = False
        load_success_target = False
        imgs_input = [None] * self.num_frames
        imgs_target = [None] * self.num_frames

        for fr in range(self.num_frames):
            if self.database[index]['input_file_' + str(fr)] is not None:
                try:
                    if not self.use_xarray:
                        img_input, _, _, _ = read_geotiff(self.database[index]['input_file_' + str(fr)]) #this will be hxwxc
                    else:
                        img_input = self.database[index]['xr_img_input_' + str(fr)]['img_input_' + str(fr)].compute().data #this will be cxhxw
                        if img_input is not None:
                            if len(img_input.shape) > 2:
                                img_input = img_input.transpose(1, 2, 0) # the result will be hxwxc
                    if img_input is not None:
                        if len(img_input.shape) == 2:
                            img_input = np.expand_dims(img_input, axis=2)

                        if img_input.shape[2] != len(self.bands):
                            print('Problem when reading {}'.format(self.database[index]['input_file_' + str(fr)]))
                            load_success_input = False
                        else:
                            img_input = self.polish_image_size(img_input, img_type='image')
                            imgs_input[fr] = img_input
                            load_success_input = True
                    else:
                        print('Problem when reading {}'.format(self.database[index]['input_file_' + str(fr)]))
                        load_success_input = False
                except:
                    print('Problem when reading {}'.format(self.database[index]['input_file_' + str(fr)]))
                    load_success_input = False
            else:
                load_success_input = False

            if self.database[index]['target_file_' + str(fr)] is not None:
                try:
                    if not self.use_xarray:
                        img_target, _, _, _ = read_geotiff(self.database[index]['target_file_' + str(fr)])
                    else:
                        img_target = self.database[index]['xr_img_target_' + str(fr)][
                            'img_target_' + str(fr)].compute().data
                        if img_target is not None:
                            if len(img_target.shape) > 2:
                                img_target = img_target.transpose(1, 2, 0) # the result will be hxwxc
                    if img_target is not None:
                        if len(img_target.shape) == 2:
                            img_target = np.expand_dims(img_target, axis=2)

                        if img_target.shape[2] != len(self.bands):
                            print('Problem when reading {}'.format(self.database[index]['target_file_' + str(fr)]))
                            load_success_target = False
                        else:
                            img_target = self.polish_image_size(img_target, img_type='image')
                            imgs_target[fr] = img_target
                            load_success_target = True
                    else:
                        print('Problem when reading {}'.format(self.database[index]['target_file_' + str(fr)]))
                        load_success_target = False
                except:
                    print('Problem when reading {}'.format(self.database[index]['target_file_' + str(fr)]))
                    load_success_target = False
            else:
                load_success_target = False

        if not load_success_input:
            imgs_input = [None] * self.num_frames

        if not load_success_target:
            imgs_target = [None] * self.num_frames

        return imgs_input, imgs_target


class CustomCollation(object):
    def __init__(self, config):
        img_size = to_2tuple(config['model_params']['img_size'])
        self.model_input_width = img_size[1]
        self.model_input_height = img_size[0]
        self.no_data_float = 0.0001 if "no_data_float" not in config["model_params"] else config["model_params"]["no_data_float"]

    def __call__(self, data):
        imgs_input = [elem['img_input'] for elem in data]
        imgs_target = [elem['img_target'] for elem in data]
        if None in imgs_target and any(isinstance(elem, torch.Tensor) for elem in imgs_target):
            print("**WARNING**! At least one of your inputs is missing target! "
                  "The target will be assumed to be a tensor filled with 0! If training/testing, then consider "
                  "removing any inputs without an explicit target from the dataset!")

        batch_size = len(imgs_input)

        widths = [int(elem.shape[2]) for elem in imgs_input]
        widths.append(self.model_input_width)
        heights = [int(elem.shape[1]) for elem in imgs_input]
        heights.append(self.model_input_height)
        max_width = np.amax(widths)
        max_height = np.amax(heights)

        channels = [int(elem.shape[3]) for elem in imgs_target if elem is not None]
        num_frames = [int(elem.shape[0]) for elem in imgs_target if elem is not None]
        if channels:
            padded_imgs_target = torch.zeros(
                (batch_size, num_frames[0], max_height, max_width, channels[0])) + self.no_data_float
            for i in range(batch_size):
                img = imgs_target[i]
                if img is not None:
                    padded_imgs_target[i, :, :int(img.shape[1]), :int(img.shape[2]), :] = img
            padded_imgs_target = padded_imgs_target.permute(0, 4, 1, 2,
                                                            3)  # batchsize, channels, num_frames, height, width
        else:
            padded_imgs_target = [None] * batch_size

        channels = [int(elem.shape[3]) for elem in imgs_input]
        num_frames_input = [int(elem.shape[0]) for elem in imgs_input if elem is not None]
        padded_imgs_input = torch.zeros(
            (batch_size, num_frames_input[0], max_height, max_width, channels[0])) + self.no_data_float
        for i in range(batch_size):
            img = imgs_input[i]
            padded_imgs_input[i, :, :int(img.shape[1]), :int(img.shape[2]), :] = img
        padded_imgs_input = padded_imgs_input.permute(0, 4, 1, 2, 3)

        new_data = {'img_target': padded_imgs_target, 'img_input': padded_imgs_input}
        for fr in range(num_frames_input[0]):
            new_data['input_file_' + str(fr)] = [elem['input_file_' + str(fr)] for elem in data]
            new_data['target_file_' + str(fr)] = [elem['target_file_' + str(fr)] for elem in data]

        return new_data
