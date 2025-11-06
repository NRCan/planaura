import random
from torch.utils.data.sampler import Sampler


class BatchSampler(Sampler):

    def __init__(self, source, batch_size, drop_last):
        super().__init__(source)
        self.source = source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.fetchers = self.create_fetcher()

    def __iter__(self):
        for fetch in self.fetchers:
            yield fetch

    def __len__(self):
        if self.drop_last:
            return len(self.source) // self.batch_size
        else:
            return (len(self.source) + self.batch_size - 1) // self.batch_size

    def create_fetcher(self):
        index_list = self.source.get_index_list()
        number_batches = self.__len__()
        batch_list = []
        for start in range(0, number_batches * self.batch_size, self.batch_size):
            batch = []
            for b in range(start, start + self.batch_size):
                batch.append(index_list[b % len(index_list)])
            batch_list.append(batch)
        return batch_list


class Sampler_NoShuffle(object):
    def return_list_index(self, database):
        database_index = [index for index in range(len(database))]
        return database_index


class Sampler_Shuffle(object):
    def return_list_index(self, database):
        database_index = [index for index in range(len(database))]
        random.shuffle(database_index)
        return database_index


def fetch_sampler(config):
    if 'sampler' in config:
        sampler_type = config['sampler']
    else:
        sampler_type = "Sampler_Shuffle"

    if sampler_type == "Sampler_Shuffle":
        return Sampler_Shuffle()
    if sampler_type == "Sampler_NoShuffle":
        return Sampler_NoShuffle()
    else:
        raise Exception("This sampler does not exist!")
