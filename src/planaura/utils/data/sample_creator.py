import torch
import numpy as np
from planaura.utils.data.sample_normalizer import SampleImageNormalizer


class DataSampleCreator(object):

    def __init__(self, config):
        num_frames = config["num_frames"]
        self.normalize_image = SampleImageNormalizer(config)
        self.to_torch = TorchFromNumpy()
        self.concatenate_frames = ConcatenateFrames(num_frames)

    def __call__(self, sample):
        sample = self.normalize_image(sample)
        sample = self.concatenate_frames(sample)
        return self.to_torch(sample)


class ConcatenateFrames(object):
    def __init__(self, num_frames):
        self.num_frames = num_frames
        # expected sample shape to the model: c*nf*height*width
        # so far, we will turn it to nf*height*width*c, and let collator take care of channel first permutation.

    def __call__(self, sample):

        to_return = dict(sample)
        imgs_target = []
        imgs_input = []
        for fr in range(self.num_frames):
            img_input, img_target = sample['img_input_' + str(fr)], sample['img_target_' + str(fr)]
            imgs_target.append(img_target)  # each one is hxwxc
            imgs_input.append(img_input)  # each one is hxwxc

        if imgs_target[0] is not None:
            img_target = np.stack(imgs_target, axis=0)
        else:
            img_target = None

        if imgs_input[0] is not None:
            img_input = np.stack(imgs_input, axis=0)
        else:
            img_input = None

        for fr in range(self.num_frames):
            to_return['img_input_' + str(fr)] = None
            to_return['img_target_' + str(fr)] = None

        to_return['img_input'] = img_input  # nf*h*w*c
        to_return['img_target'] = img_target  # nf*h*w*c
        return to_return


class TorchFromNumpy(object):
    def __call__(self, sample):
        img_input, img_target = sample['img_input'], sample['img_target']
        returned_sample = dict(sample)
        if img_input is None:
            raise ValueError('You cannot run the model without input')
        returned_sample['img_input'] = torch.from_numpy(img_input)
        if img_target is not None:
            returned_sample['img_target'] = torch.from_numpy(img_target)

        return returned_sample
