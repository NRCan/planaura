import copy
import numpy as np

class SampleImageNormalizer(object):

    def __init__(self, config):

        self.mean = np.array([[config["model_params"]["data_mean"]]])
        self.std = np.array([[config["model_params"]["data_std"]]])
        self.num_frames = config["num_frames"]
        self.no_data = config["model_params"]["no_data"]
        self.no_data_float = 0.0001 if "no_data_float" not in config["model_params"] else config["model_params"]["no_data_float"]

    def __call__(self, sample):

        returned_sample = copy.deepcopy(sample)

        # for now, for the reconstruction task, the target will be built at loss calculation. So, no action to take on target.
        for i in range(self.num_frames):
            imageg = returned_sample['img_input_' + str(i)]
            if imageg is not None:  # size of imageg: ( H, W, C)
                imageg = imageg.astype(np.float32)
                imageg = np.where(imageg == self.no_data, self.no_data_float, (imageg - self.mean) / self.std)
                returned_sample['img_input_' + str(i)] = imageg

        return returned_sample


class SampleImageUnNormalizer(object):

    def __init__(self, config):
        self.mean = np.array([[config["model_params"]["data_mean"]]])
        self.std = np.array([[config["model_params"]["data_std"]]])
        self.no_data = config["model_params"]["no_data"]
        self.no_data_float = 0.0001 if "no_data_float" not in config["model_params"] else config["model_params"]["no_data_float"]

    def __call__(self, np_prediction):
        # np_prediction:  image of size ( H, W, C) to be un-normalized.
        # C = np_prediction.shape[2]
        np_prediction = np_prediction.astype(np.float32)
        np_prediction = np.where(np_prediction == self.no_data_float, self.no_data,
                                 (np_prediction * self.std) + self.mean)

        return np_prediction
