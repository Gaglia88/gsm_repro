import traceback
import json
import pandas as pd

class Utils(object):

    @staticmethod
    def load_datasets(path='/home/app/datasets/datasets.json', dtype=''):
        f = open(path)
        datasets = json.load(f)
        f.close()
        if len(dtype) > 0:
            datasets = list(filter(lambda d: d['type']==dtype, datasets))
        return datasets
        
    @staticmethod
    def get_train_test(df, n_samples, label, seed, pos_value = 1, neg_value = 0):
        #Extracts the matching records
        match = df[df[label]==pos_value]
        #Extracts the non-matching records
        non_match = df[df[label]==neg_value]

        #Takes n/2 samples from the matching records
        pos_train = match.sample(n=int(n_samples/2), replace=False, random_state=seed)
        #Removes the extracted samples, these are used for testing
        pos_test = match.drop(index=pos_train.index)
        #To have a balanced training set, takes the same number of samples of the matching records
        neg_train = non_match.sample(n=len(pos_train), replace=False, random_state=seed)
        #Removes the extracted samples, these are used for testing
        neg_test = non_match.drop(index=neg_train.index)

        #Training set
        train_norm = pd.concat([pos_train, neg_train], axis=0)
        X_train = train_norm.drop(label, axis=1)
        y_train = train_norm[[label]]

        #Test set
        test = pd.concat([pos_test, neg_test], axis=0)
        X_test = test.drop(label, axis=1)
        y_test = test[[label]]
        
        return X_train, X_test, y_train, y_test

    @staticmethod
    def split_train_test(df, train_set_size):
        # n_samples: Number of samples per class (actually spark do not ensure this exact value during sampling)
        n_samples = train_set_size/2

        # Sampling of matching pairs
        matches = df.where("is_match == 1")
        m_n = n_samples/matches.count()
        m_train, m_test = matches.randomSplit([m_n, 1-m_n])

        # Sampling of non-matching pairs
        non_matches = df.where("is_match == 0")
        nm_n = n_samples/non_matches.count()
        nm_train, nm_test = non_matches.randomSplit([nm_n, 1-nm_n])

        # Train/Test
        train = m_train.union(nm_train)
        test = m_test.union(nm_test)
        return train, test

    @staticmethod
    def load_data(dataset_data, base_folder=""):
        import sparker
        """
        Load a dataset
        """
        try:
            profiles1 = None
            profiles2 = None
            profiles = None
            new_gt = None
            separator_ids = None
            
            if dataset_data['format'] == 'json':
                # Profiles contained in the first dataset
                profiles1 = sparker.JSONWrapper.load_profiles(base_folder+dataset_data['base_path']+"/"+dataset_data['d1'], 
                                                              real_id_field = dataset_data['d1_id_field'])
            else:
                profiles1 = sparker.CSVWrapper.load_profiles(base_folder+dataset_data['base_path']+"/"+dataset_data['d1'], 
                                                              real_id_field = dataset_data['d1_id_field'])

            # Loads the groundtruth, takes as input the path of the file and the names of the attributes that represent
            # respectively the id of profiles of the first dataset and the id of profiles of the second dataset
            if dataset_data['format'] == 'json':
                gt = sparker.JSONWrapper.load_groundtruth(base_folder+dataset_data['base_path']+"/"+dataset_data['gt'], 
                                                          dataset_data['gt_d1_field'],
                                                          dataset_data['gt_d2_field'])
            else:
                gt = sparker.CSVWrapper.load_groundtruth(base_folder+dataset_data['base_path']+"/"+dataset_data['gt'], 
                                                          dataset_data['gt_d1_field'],
                                                          dataset_data['gt_d2_field'])
                
            if dataset_data['type'] == 'clean':
                # Max profile id in the first dataset, used to separate the profiles in the next phases
                separator_id = profiles1.map(lambda profile: profile.profile_id).max()
                # Separators, used during blocking to understand from which dataset a profile belongs. It is an array because sparkER
                # could work with multiple datasets
                separator_ids = [separator_id]
            
                if dataset_data['format'] == 'json':
                    # Profiles contained in the first dataset
                    profiles2 = sparker.JSONWrapper.load_profiles(base_folder+dataset_data['base_path']+"/"+dataset_data['d2'], 
                                                                  start_id_from = separator_id+1, 
                                                                  real_id_field = dataset_data['d2_id_field'])
                else:
                    profiles2 = sparker.CSVWrapper.load_profiles(base_folder+dataset_data['base_path']+"/"+dataset_data['d2'], 
                                                                 start_id_from = separator_id+1, 
                                                                 real_id_field = dataset_data['d2_id_field'])
                
                profiles = profiles1.union(profiles2)
                    
                # Max profile id
                max_profile_id = profiles2.map(lambda profile: profile.profile_id).max()

                # Converts the groundtruth by replacing original IDs with those given by Spark
                new_gt = sparker.Converters.convert_groundtruth(gt, profiles1, profiles2)
            else:
                profiles = profiles1
                    
                # Max profile id
                max_profile_id = profiles.map(lambda profile: profile.profile_id).max()

                # Converts the groundtruth by replacing original IDs with those given by Spark
                new_gt = sparker.Converters.convert_groundtruth(gt, profiles)
            
            return profiles1, profiles2, profiles, new_gt, max_profile_id, separator_ids
        except Exception:
            print('Cannot load the data')
            print(traceback.format_exc())
            

    @staticmethod
    def blocking_cleaning(profiles, separator_ids, pf=1.0, ff=0.8):
        import sparker
        """
        Performs the token blocking and the cleaning.
        pf: block purging factor
        ff: block filtering factor
        """
        try:
            blocks = sparker.Blocking.create_blocks(profiles, separator_ids)
            #print("Number of blocks",blocks.count())
            
            # Perfoms the purging
            blocks_purged = sparker.BlockPurging.block_purging(blocks, pf)
            
            # Performs the cleaning
            (profile_blocks, profile_blocks_filtered, blocks_after_filtering) = sparker.BlockFiltering.block_filtering_quick(blocks_purged, ff, separator_ids)
            
            return (profile_blocks, profile_blocks_filtered, blocks_after_filtering)
        except Exception:
            print('Cannot perform the blocking')
            print(traceback.format_exc())
    