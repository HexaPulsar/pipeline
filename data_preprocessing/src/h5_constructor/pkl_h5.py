import glob
import os
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm
import joblib
import h5py
import pickle
import pandas as pd
import logging


class PKL2H5:
    def __init__(self,
                 input_directory: str,  # TODO add mutliple directory functionality
                 output_directory: str,
                 cross_cont_check_path: str,
                 output_h5_filename: str,
                 assert_path_type: bool = False,  # TODO check that all pkl files are astroobjects
                 h5_file_dirname: str = 'h5_dataset',
                 metadata_dirname: str = 'metadata',
                 features_dirname: str = 'features',
                 logger=None,
                 purge_output_directory: bool = False,
                 **kwargs
                 ):
        """_summary_
        #TODO rellenar docstring
        Args:
            input_directory (str): _description_
            cross_cont_check_path (str): _description_
            output_h5_filename (str): _description_
            assert_path_type (bool, optional): _description_. Defaults to False.
            metadata_dirname (str, optional): _description_. Defaults to 'metadata'.
            features_dirname (str, optional): _description_. Defaults to 'features'.
            logger (_type_, optional): _description_. Defaults to None.
            purge_output_directory (bool, optional): _description_. Defaults to False.
        """
        self.input_directory = input_directory
        self.cross_contamination_path = cross_cont_check_path
        self.h5_name = output_h5_filename
        # TODO assert objecttypes
        # TODO try except for output directory and assert directories exist
        self.output_directory = output_directory

        if all([os.path.exists(self.output_directory),purge_output_directory]):
            answer = input(
                f'{self.output_directory} exists and purge is set to {purge_output_directory}. Would you like to purge the directory? This will erase everything inside {self.output_directory}\n y/n:')
            if answer.lower() == 'y':
                answer = input(f'Are you sure? \n y/n:')
                if answer.lower() == 'y':
                    self._purge_directory()
            elif answer.lower() == 'n':
                exit()


        self.h5_output, self.md_output, self.ft_output = self._init_dirs(output_directory,
                                                                        h5_file_dirname,
                                                                        metadata_dirname,
                                                                        features_dirname)
        if logger is None:
            self.logger = self._init_logger(output_directory,
                                           'h5_creator.log')
        else:
            self.logger = logger
        self.logger.info(f" directories @ {self.output_directory}")
        self.filepath_list = self._get_pkl_filepaths()

    def create_dataset(self, train_val_split: float,
                       k_folds: int,
                       generate_features_qt: bool,
                       generate_metadata_qt: bool,
                       metadata_prefix: str = 'metadata_qt',
                       features_prefix: str = 'features_qt',):
        """_summary_
        #TODO generate docstring
        Args:
            train_val_split (float): _description_
            k_folds (int): _description_
            generate_features_qt (bool): _description_
            generate_metadata_qt (bool): _description_
            metadata_prefix (str, optional): _description_. Defaults to 'metadata_qt'.
            features_prefix (str, optional): _description_. Defaults to 'features_qt'.
        """
        incoming_aos = self._get_oids()
        cross_cont_df = pd.read_parquet(self.cross_contamination_path)
        #print(cross_cont_df['oid'])
        not_contaminated_df = self._check_cross_contamination(incoming_aos,
                                                             cross_cont_df)
         
        self._initialize_h5_keys(not_contaminated_df, train_val_split, k_folds)
        self.keys = ['flux','time','mask','md_cols','ft_cols'] 
        self._init_dataset_arrays(self.keys,
                                  lc_array_shape=(len(self.filepath_list),200,2),
                                  ft_array_shape=(len(self.filepath_list),187,1),
                                  md_array_shape=(len(self.filepath_list),6,1))
        
        #self._get_lightcurve_data_arrays()

        if generate_features_qt:
            self._get_ft()
            self.logger.info(
                f'Creating features Quantile Transformation for {k_folds} folds')
            for i in range(k_folds):
                self._create_ft_fold(seed=i, prefix=features_prefix)
                
        if generate_metadata_qt:
            self._get_md()
            self.logger.info(
                f'Creating metadata Quantile Transformation for {k_folds} folds')
            for i in range(k_folds):
                self._create_md_fold(seed=i, prefix=metadata_prefix)
        # TODO add final key print
        # TODO report statistics
        # TODO report datatypes
        # TODO assert array shapes

    def _init_logger(self, output_directory, log_file_name):
        logFormatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s]: %(message)s", datefmt="%H:%M")
        logger = logging.getLogger()
        logging.basicConfig(filename=log_file_name,
                            encoding='utf-8',
                            level=logging.DEBUG,
                            filemode='w')
        fileHandler = logging.FileHandler(
            "%s/%s" % (output_directory, log_file_name))
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)
        logger.propagate = False
        return logger

    def _init_dirs(self, output_directory, h5_file_dirname, metadata_dirname, features_dirname):
        h5_output = os.path.join(output_directory, h5_file_dirname)
        md_output = os.path.join(output_directory, metadata_dirname)
        ft_output = os.path.join(output_directory, features_dirname)

        try:
            # make home dir @ output_directory
            os.makedirs(self.output_directory)
            # make subdirectories for h5 and quantiles
            try:
                os.makedirs(h5_output)
            except RuntimeError as e:
                print(f"Runtime error: {e}")
            try:
                os.makedirs(md_output)
            except RuntimeError as e:
                print(f"Runtime error: {e}")
            try:
                os.makedirs(ft_output)
            except RuntimeError as e:
                print(f"Runtime error: {e}")
            return h5_output, md_output, ft_output
        except RuntimeError as e:
            print(f"Directory already exists.")
        return h5_output, md_output, ft_output

    def _init_dataset_arrays(self, keys,
                             lc_array_shape,
                             ft_array_shape,
                             md_array_shape):
        """init the arrays so objects are inserted directly into the h5 file instead of preserving all of them in memory and then inserting

        Args:
            keys (_type_): _description_
            n_elements (_type_): _description_
            row_length (_type_): _description_
            column_length (_type_): _description_
        """
        for key in keys:
            if key == 'ft_cols':
                
                empty_dataset = np.empty(ft_array_shape)
            if key == 'md_cols':
                empty_dataset = np.empty(md_array_shape)
            else:
                empty_dataset = np.empty(lc_array_shape)
            with h5py.File(os.path.join(self.h5_output, self.h5_name), 'r+') as f:
                print(key,empty_dataset.shape)
                f.create_dataset(key, data=empty_dataset)
                assert key in list(f.keys())
                self.logger.debug(
                    f"Added key '{key}' dataset shape {empty_dataset.shape} as a HDF5 root level key")
            print(
                    f"Added key '{key}'  dataset shape {empty_dataset.shape} as a HDF5 root level key")
 


    def _get_lightcurve_data_arrays(self):
        self._get_flux()
        self._get_mask()
        self._get_time()

    def _purge_directory(self):
        # TODO purge directory
        pass

    def _get_pkl_filepaths(self):
        self.logger.info(f'Searching directory path: {self.input_directory}')
        return glob.glob(f'{self.input_directory}*')

    def _get_oids(self):
        """ Get OIDs from all the astro objects present in the directory provided.
        """
        grab_oids = []
        #TODO: this operation could be paralelized
        for dir in tqdm(self.filepath_list,
                        desc="Searching for .pkl files",
                        total=len(self.filepath_list), unit='files'):
            with (open(dir, "rb")) as openfile:
                load = pickle.load(openfile)
                grab_oids.append(load['oid'])
        self.logger.info(f'Found {len(self.filepath_list)} .pkl files')
        return pd.DataFrame({'oid': grab_oids}, dtype='str')

    def _check_directory_objects(self):
        # TODO a function to check that all objects that will be loaded
        # by the dataset constructors are of type AstroObject
        pass

    def _check_cross_contamination(self, incoming_oids_df, crosscheck_df):
        """Check for cross contamination between directory OIDs and a provided dataframe with an OIDs column
        """
        try:
            # TODO replace direct parquet file loading for input argument of type list
            ff_oids = crosscheck_df['oid']

            cross_contamination = pd.merge(ff_oids, incoming_oids_df, on='oid',
                                           how='inner')['oid'].unique()
            self.logger.debug(f'{cross_contamination.shape}')
            incoming_oids_df = incoming_oids_df[~incoming_oids_df['oid'].isin(
                cross_contamination)]
            n_removed = 'TODO'
            # TODO add n_removed for logging
            self.logger.info(
                f'Removed {n_removed} objects from future dataset')
            # TODO sanity check

           # print(cross_contamination)
            return incoming_oids_df
        except FileNotFoundError:
            self.logger.error(
                f"Cross contamination file not found at {self.cross_contamination_path}")
        except pd.errors.EmptyDataError:
            self.logger.error(
                "Cross contamination file exists but contains no data")
        except KeyError as e:
            self.logger.error(f"Missing required column in dataframe: {e}")
        except Exception as e:
            self.logger.error(
                f"Unexpected error during cross contamination check: {e}")

    def _get_flux(self): 
        key_name = 'flux'
        for i in tqdm(range(len(self.filepath_list)), desc='   - Getting flux data for all astro objects', total=len(self.filepath_list)):
            
            with open(self.filepath_list[i], "rb") as openfile:
                load = pickle.load(openfile)
                #grab_flux.append(load[key_name])

            with h5py.File(os.path.join(self.h5_output, self.h5_name), 'r+') as f:
                try:
                    f.get(key_name)[i,:,:] = load
                except Exception as e:
                    self.logger.error(
                        f"Unexpected error: {e}")

    def _get_mask(self): 
        key_name = 'mask'
        for i in tqdm(range(len(self.filepath_list)), desc=f'   - Getting {key_name} data for all astro objects', total=len(self.filepath_list)):
            
            with open(self.filepath_list[i], "rb") as openfile:
                load = pickle.load(openfile)
                #grab_flux.append(load[key_name])

            with h5py.File(os.path.join(self.h5_output, self.h5_name), 'r+') as f:
                try:
                    f.get(key_name)[i,:,:] = load.astype(bool)
                except Exception as e:
                    self.logger.error(
                        f"Unexpected error: {e}")

    def _get_time(self): 
        key_name = 'time'
        for i in tqdm(range(len(self.filepath_list)), desc=f'   - Getting {key_name} data for all astro objects', total=len(self.filepath_list)):
            
            with open(self.filepath_list[i], "rb") as openfile:
                load = pickle.load(openfile)
                #grab_flux.append(load[key_name])

            with h5py.File(os.path.join(self.h5_output, self.h5_name), 'r+') as f:
                try:
                    f.get(key_name)[i,:,:] = load
                except Exception as e:
                    self.logger.error(
                        f"Unexpected error: {e}")

    def _get_ft(self):
        # TODO check!
        key_name = 'ft_cols'
        for i in tqdm(range(len(self.filepath_list)), desc=f'   - Getting {key_name} data for all astro objects', total=len(self.filepath_list)):
            with (open(self.filepath_list[i], "rb")) as openfile:
                load = pickle.load(openfile)
                print(load['ft_cols'])
                print(load['md_cols'])
                feats = pd.DataFrame(load['ft_cols'].value.values)
                print(feats.shape)
                feats = feats.replace([np.inf, -np.inf], np.nan)
                #grab_ft.append(feats.values)
            with h5py.File(os.path.join(self.h5_output, self.h5_name), 'r+') as f:
                # Print original dictionary keys
                f.get('ft_cols')[i,:,:] =np.array(feats.values).astype(float)
                # Print HDF5 root level keys
                # TODO assert that key was inserted
                # assert inserted dataset shape matches for loop length
            pass 

    def _get_md(self):
        # TODO check!
        #grab_md = []
        key_name = 'md_cols'
        for i in tqdm(range(len(self.filepath_list)), desc=f'   - Getting {key_name} data for all astro objects', total=len(self.filepath_list)):
            with (open(self.filepath_list[i], "rb")) as openfile:
                load = pickle.load(openfile)
                feats = pd.DataFrame(load['md_cols'].value.values)
                feats = feats.replace([np.inf, -np.inf], np.nan)
                #grab_ft.append(feats.values)
            with h5py.File(os.path.join(self.h5_output, self.h5_name), 'r+') as f:
                # Print original dictionary keys
                f.get('md_cols')[i,:,:] =np.array(feats.values).astype(float)
                # Print HDF5 root level keys
                # TODO assert that key was inserted
                # assert inserted dataset shape matches for loop length
            pass

    def _create_single_train_val_split(self, df_oids: pd.DataFrame, split_frac: float):
        """Will create a train/validation split by splitting a pd dataframe of oids (single column dataframe where index is range of N and column contains the oids)

        Args:
            df_oids (pd.DataFrame): _description_
        """
        try:
            if not (0 < split_frac < 1):
                raise ValueError(
                    "split_frac must be between 0 and 1 (exclusive)")
            if df_oids.empty:
                raise ValueError("df_oids must not be empty")
        except ValueError as e:
            raise e  # Re-raising the exception with the original message
        validation = df_oids.sample(
            frac=split_frac, random_state=42)  # for reproducibility

        train = df_oids.loc[~df_oids.index.isin(validation.index)]
        return train, validation

    def _initialize_h5_keys(self, oids_df: pd.DataFrame, val_frac: float, k_folds: int):
        """
        Create an HDF5 file and save data from a dictionary.

        Parameters:
        filename (str): Name or path of the HDF5 file to create
        data_dict (dict): Dictionary containing the data to save
        """
        try:
            if k_folds == 0:
                raise ValueError("k_folds must be a nonzero integer")
            if not isinstance(val_frac, float):
                raise TypeError("val_frac must be a float")
            if not (0 < val_frac < 1):
                raise ValueError(
                    "val_frac must be between 0 and 1 (exclusive)")
            if not isinstance(k_folds, int):
                raise TypeError("k_folds must be an integer")
        except (TypeError, ValueError) as e:
            raise e

        keys2insert = {}
        self.logger.info("Creating %s folds for train/val split %s/%s" %
                         (k_folds, int((1-val_frac)*100), int(val_frac*100)))
        for i in range(k_folds):
            train, val = self._create_single_train_val_split(oids_df, 0.2)
            train = train.index.tolist()
            train.sort()
            val = val.index.tolist()
            val.sort()

            # assert not np.any(np.in1d(train.values.flatten(), val.values.flatten()))
            keys2insert.update({f'train_{i}': train})
            keys2insert.update({f'validation_{i}': val})

        with h5py.File(os.path.join(self.h5_output, self.h5_name), 'w') as f:
            for key, value in keys2insert.items():
                if not isinstance(value, np.ndarray):
                    value = np.array(value)
                f.create_dataset(key, data=value)
        # TODO test that the correct keys have been inserted
        # log info and debug

    def _create_ft_fold(self, seed, prefix=''):
        qt = QuantileTransformer(n_quantiles=1000,
                                 random_state=0,
                                 output_distribution="uniform"
                                 )
        with h5py.File(os.path.join(self.h5_output, self.h5_name), 'r') as f:
            these_idx = f.get(f'train_{seed}')
            feats = f.get('ft_cols')[these_idx]
        feats = feats.reshape(feats.shape[0], feats.shape[1])
        nan_mask = np.isnan(np.array(feats))
        df = pd.DataFrame(feats)
        qt.fit(df[~nan_mask])
        # df = qt.transform(df.fillna(12345)) + 0.1
        # df[nan_mask] = 0.0
        # df = df.reshape(df.shape[0],df.shape[1],1)
        # feats = np.concatenate([feat for feat in collect_feats])
        # print(feats)
        
        joblib.dump(qt, os.path.join(
            self.ft_output, f'{prefix}_{seed}.joblib'))

    def _create_md_fold(self, seed, prefix=''):
        qt = QuantileTransformer(n_quantiles=1000,
                                 random_state=0,
                                 output_distribution="uniform")
        with h5py.File(os.path.join(self.h5_output, self.h5_name), 'r') as f:
            these_idx = f.get(f'train_{seed}')
            feats = f.get('md_cols')[these_idx]
        feats = feats.reshape(feats.shape[0], feats.shape[1])
        nan_mask = np.isnan(np.array(feats))
        df = pd.DataFrame(feats)
        qt.fit(df[~nan_mask])
        

        joblib.dump(qt, os.path.join(
            self.md_output, f'{prefix}_{seed}.joblib'))
