from planaura.utils.data.data_folder_parser import DataFolderParser


def fetch_parser(config, mode='train'):
    dataset_name = "reconstruction"
    im_formats = config["image_format"] if "image_format" in config else None
    if im_formats is None:
        im_formats = 'tif'
    ignore_strings_input = None
    ignore_strings_target = None
    if "search_input_folder_ignore" in config:
        ignore_strings_input = config["search_input_folder_ignore"]
    if "search_target_folder_ignore" in config:
        ignore_strings_target = config["search_target_folder_ignore"]

    use_xarray = False if "use_xarray" not in config else config["use_xarray"]
    paths_included_in_csvs = False if "paths_included_in_csvs" not in config else config["paths_included_in_csvs"]

    if dataset_name == "reconstruction":
        if mode == 'train':
            data_input_folders_list = [''] * config['num_frames']
            data_target_folders_list = [''] * config['num_frames']
            mask_folders_list = [''] * config['num_frames']
            if not paths_included_in_csvs:
                for fr in range(config['num_frames']):
                    data_input_folders_list[fr] = config["train_input_folder_frame_" + str(fr)]
                    st = "train_target_folder_frame_" + str(fr)
                    t = '' if st not in config else config[st]
                    data_target_folders_list[fr] = t

            return DataFolderParser(config,
                                    data_input_folders_list, data_target_folders_list,
                                    im_formats=im_formats,
                                    ignore_strings_input=ignore_strings_input,
                                    ignore_strings_target=ignore_strings_target,
                                    use_xarray=use_xarray)
        elif mode == 'test':
            data_input_folders_list = [''] * config['num_frames']
            data_target_folders_list = [''] * config['num_frames']
            mask_folders_list = [''] * config['num_frames']
            if not paths_included_in_csvs:
                for fr in range(config['num_frames']):
                    data_input_folders_list[fr] = config["test_input_folder_frame_" + str(fr)]
                    st = "test_target_folder_frame_" + str(fr)
                    t = '' if st not in config else config[st]
                    data_target_folders_list[fr] = t

            return DataFolderParser(config,
                                    data_input_folders_list, data_target_folders_list,
                                    im_formats=im_formats,
                                    ignore_strings_input=ignore_strings_input,
                                    ignore_strings_target=ignore_strings_target,
                                    use_xarray=use_xarray)
        elif mode == 'inference':
            data_input_folders_list = [''] * config['num_frames']
            data_target_folders_list = [''] * config['num_frames']
            if not paths_included_in_csvs:
                for fr in range(config['num_frames']):
                    data_input_folders_list[fr] = config["inference_input_folder_frame_" + str(fr)]
            return DataFolderParser(config,
                                    data_input_folders_list, data_target_folders_list,
                                    im_formats=im_formats,
                                    ignore_strings_input=ignore_strings_input,
                                    ignore_strings_target=ignore_strings_target,
                                    use_xarray=use_xarray)
        else:
            raise Exception("invalid mode {}".format(mode))
    else:
        raise Exception("invalid dataset name {}  for parsing".format(dataset_name))


def fetch_parser_geotiff(config, mode='inference'):
    dataset_name = "reconstruction"
    im_formats = config["image_format"] if "image_format" in config else None
    paths_included_in_csvs = False if "paths_included_in_csvs" not in config else config["paths_included_in_csvs"]
    if im_formats is None:
        im_formats = 'tif'
    ignore_strings_input = None
    ignore_strings_target = None
    if "search_input_folder_ignore" in config:
        ignore_strings_input = config["search_input_folder_ignore"]
    if "search_target_folder_ignore" in config:
        ignore_strings_target = config["search_target_folder_ignore"]

    if dataset_name == "reconstruction":
        if mode == 'inference':
            data_input_folders_list = [''] * config['num_frames']
            data_target_folders_list = [''] * config['num_frames']
            if not paths_included_in_csvs:
                for fr in range(config['num_frames']):
                    data_input_folders_list[fr] = config["inference_input_folder_geotiff_frame_" + str(fr)]

            return DataFolderParser(config,
                                    data_input_folders_list, data_target_folders_list,
                                    im_formats=im_formats,
                                    ignore_strings_input=ignore_strings_input,
                                    ignore_strings_target=ignore_strings_target,
                                    use_xarray=False)
        else:
            raise Exception("invalid mode {} is not supported in geotiff processing".format(mode))
    else:
        raise Exception("invalid dataset name {}  for parsing".format(dataset_name))
