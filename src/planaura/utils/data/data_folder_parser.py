import collections
import os
import gc
import copy
import time
import xarray
import concurrent.futures
import pandas as pd
from timm.models.layers import to_2tuple


class DataFolderParser(object):

    def __init__(self, config, data_input_folders_list, data_target_folders_list,
                 im_formats=None, ignore_strings_input=None, ignore_strings_target=None,
                 use_xarray=False):
        self.num_frames = config['num_frames']
        self.data_input_folders_list = data_input_folders_list
        self.data_target_folders_list = data_target_folders_list
        self.data_base = []
        self.im_formats = im_formats
        if im_formats is None:
            self.im_formats = 'tif'
        self.bands = config["model_params"]["bands"]
        img_size = to_2tuple(config['model_params']['img_size'])
        self.model_input_width = img_size[1]
        self.model_input_height = img_size[0]
        self.ignore_strings_input = ignore_strings_input
        self.ignore_strings_target = ignore_strings_target
        if self.ignore_strings_target is None:
            self.ignore_strings_target = []
        if self.ignore_strings_input is None:
            self.ignore_strings_input = []
        self.use_xarray = use_xarray
        self.config = config


    def read_data(self, data_file, path_included_in_csv=False):

        try:
            self._read_data(data_file, path_included_in_csv)
        except Exception as e:
            print(e)
            if path_included_in_csv:
                raise ValueError("As you set path_included_in_csv to True, you need to create the CSV files yourself!")
            try:
                print('CSV data file does not exist; trying to create the csv data file ...')
                r = self._read_folder(data_file)
                if r == 0:
                    raise ValueError('No input data available!')
            except Exception as e:
                print(e)
                raise ValueError('Invalid CSV data file: {}: {}'.format(data_file, e))


    def _read_folder(self, save_at_data_file):
        column_name = {}
        data_input = []
        data_target = []
        all_input_files_unique = []
        for fr in range(self.num_frames):
            column_name.update({'input_file_' + str(fr): [], 'target_file_' + str(fr): []})  # The name of the columns

        for fr in range(self.num_frames):
            if self.data_input_folders_list[fr] != '':
                if not os.path.exists(self.data_input_folders_list[fr]):
                    os.makedirs(self.data_input_folders_list[fr])

        tmp = []
        unique_input_versus_repeats = {}
        for fr in range(self.num_frames):
            if self.data_input_folders_list[fr] != '':
                for sub in os.listdir(self.data_input_folders_list[fr]):
                    f = os.path.join(self.data_input_folders_list[fr], sub)
                    if os.path.isfile(f):
                        if f.lower().endswith(self.im_formats):
                            original_sub = sub
                            for remove_string in self.ignore_strings_input:
                                sub = sub.replace(remove_string, '')
                            tmp.append(sub)
                            if sub in unique_input_versus_repeats.keys():
                                unique_input_versus_repeats[sub].append(original_sub)
                            else:
                                unique_input_versus_repeats[sub] = [original_sub]

        counter = dict(collections.Counter(tmp))
        for key, val in counter.items():
            if val == self.num_frames:
                all_input_files_unique.append(key)

        if not all_input_files_unique:
            return 0

        for fr in range(self.num_frames):
            if self.data_target_folders_list[fr] != '':
                if not os.path.exists(self.data_target_folders_list[fr]):
                    os.makedirs(self.data_target_folders_list[fr])

        tmp = []
        unique_target_versus_repeats = {}
        for fr in range(self.num_frames):
            if self.data_target_folders_list[fr] != '':
                for sub in os.listdir(self.data_target_folders_list[fr]):
                    f = os.path.join(self.data_target_folders_list[fr], sub)
                    if os.path.isfile(f):
                        if f.lower().endswith(self.im_formats):
                            original_sub = sub
                            for remove_string in self.ignore_strings_target:
                                sub = sub.replace(remove_string, '')
                            tmp.append(sub)
                            if sub in unique_target_versus_repeats.keys():
                                unique_target_versus_repeats[sub].append(original_sub)
                            else:
                                unique_target_versus_repeats[sub] = [original_sub]

        counter = dict(collections.Counter(tmp))
        tmp_target_unique = []
        for key, val in counter.items():
            if val == self.num_frames:
                if key in all_input_files_unique:
                    tmp_target_unique.append(key)

        for input_item in all_input_files_unique:
            data_input.append(unique_input_versus_repeats[input_item])
            if input_item in tmp_target_unique:
                data_target.append(unique_target_versus_repeats[input_item])
            else:
                data_target.append([None] * len(unique_input_versus_repeats[input_item]))

        try:
            for fr in range(self.num_frames):
                column_name['input_file_' + str(fr)] = [data_input[j][fr] for j in range(len(data_input))]
                column_name['target_file_' + str(fr)] = [data_target[j][fr] for j in range(len(data_target))]
            df = pd.DataFrame.from_dict(column_name)
            df.to_csv(save_at_data_file)
        except:
            raise ValueError('invalid CSV data file: {}'.format(save_at_data_file))

        self.read_data(save_at_data_file, False)

        return 1

    def _read_data(self, csv_fullpath, path_included_in_csv):

        df_im = pd.read_csv(csv_fullpath)
        column_headers = list(df_im.columns.values)
        if not (
                "input_file_0" in column_headers
        ):
            raise ValueError(
                "The csv file should at least contain the columns of input_file_0"
            )

        look_for_target = False
        if 'target_file_0' in column_headers:
            look_for_target = True

        all_database_items = []
        all_input_files = []
        all_target_files = []
        start = time.time()
        for index, row in df_im.iterrows():
            input_files = [None] * self.num_frames
            target_files = [None] * self.num_frames

            for fr in range(self.num_frames):
                if not pd.isnull(row['input_file_' + str(fr)]) and row['input_file_' + str(fr)] is not None:
                    if not path_included_in_csv:
                        input_files[fr] = os.path.join(self.data_input_folders_list[fr], row['input_file_' + str(fr)])
                    else:
                        input_files[fr] = row['input_file_' + str(fr)]
                if look_for_target:
                    if not pd.isnull(row['target_file_' + str(fr)]) and row['target_file_' + str(fr)] is not None:
                        if not path_included_in_csv:
                            target_files[fr] = os.path.join(self.data_target_folders_list[fr],
                                                            row['target_file_' + str(fr)])
                        else:
                            target_files[fr] = row['target_file_' + str(fr)]

            inputs_sane = True
            for fr in range(self.num_frames):
                if input_files[fr] is None:
                    inputs_sane = False
                    break

            if inputs_sane:
                new_database_item = {'im_target': None, 'img_input': None}
                all_database_items.append(new_database_item)
                all_input_files.append(input_files)
                all_target_files.append(target_files)

        if not self.use_xarray:
            for fs in range(len(all_database_items)):
                new_database_item = copy.deepcopy(all_database_items[fs])
                input_files = copy.deepcopy(all_input_files[fs])
                target_files = copy.deepcopy(all_target_files[fs])
                for fr in range(self.num_frames):
                    new_database_item.update({
                        'input_file_' + str(fr): input_files[fr],
                        'target_file_' + str(fr): target_files[fr],
                        'img_input_' + str(fr): None,
                        'img_target_' + str(fr): None,
                        'xr_img_input_' + str(fr): None,
                        'xr_img_target_' + str(fr): None
                    })
                self.data_base.append(new_database_item)
        else:
            try:
                cpu_count = os.cpu_count()
                print(f'CPU count = {cpu_count}')
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Submit the tasks for all frames
                    xarray_features = [executor.submit(self._open_xarray_rows,
                                                       all_database_items[fs],
                                                       all_input_files[fs],
                                                       all_target_files[fs])
                                       for fs in range(len(all_database_items))]

                    for future in concurrent.futures.as_completed(xarray_features):
                        new_database_item = future.result()
                        self.data_base.append(new_database_item)
                        # Garbage collection to free up memory
                        gc.collect()

            except Exception as e:
                print(e)
                raise ValueError("Something went wrong when loading data to x-array")


    def _open_xarray_rows(self, new_database_item, input_files, target_files):
        for fr in range(self.num_frames):
            xr_input = xarray.open_dataset(input_files[fr], chunks={"band": len(self.bands),
                                                                    "y": self.model_input_height,
                                                                    "x": self.model_input_width},
                                           mask_and_scale=False)
            data_variable_name = list(xr_input.data_vars.keys())[0]
            xr_input = xr_input.rename({data_variable_name: 'img_input_' + str(fr)})

            if target_files[fr] is not None:
                xr_target = xarray.open_dataset(target_files[fr], chunks={"band": len(self.bands),
                                                                          "y": self.model_input_height,
                                                                          "x": self.model_input_width},
                                                mask_and_scale=False)
                data_variable_name = list(xr_target.data_vars.keys())[0]
                xr_target = xr_target.rename({data_variable_name: 'img_target_' + str(fr)})
            else:
                xr_target = None

            # Update the database item with processed data
            new_database_item.update({
                'input_file_' + str(fr): input_files[fr],
                'target_file_' + str(fr): target_files[fr],
                'xr_img_input_' + str(fr): copy.deepcopy(xr_input),
                'xr_img_target_' + str(fr): copy.deepcopy(xr_target),
                'img_input_' + str(fr): None,
                'img_target_' + str(fr): None
            })

            # Close and clean up the datasets
            xr_input.close()
            del xr_input
            if xr_target is not None:
                xr_target.close()
                del xr_target

        return new_database_item
