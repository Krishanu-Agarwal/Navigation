import os
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset

def create_data(datasets_dict,  data_folder_dict, out_gt_num, preds, mode):
    print(f"Loading Dataset: {mode}")
    dataset = {
        'src': [],
        'trg': [],
        'seq_start': [],
        'frames': [],
        'dataset': [],
        'peds': [],
        'dataset_name': data_folder_dict
    }
    src_data = []
    trg_data = []
    frame_data = []
    seq_start_data = []
    ped_ids = []
    data_num = []
    for i, data in enumerate(datasets_dict):
        print(f"{i+1}/{len(datasets_dict)}  - loading {data}")
        raw_data = pd.read_csv(os.path.join(data_folder_dict, data), delimiter ='\t',
                                                names=["frame", "ped", "x", "y"],usecols=[0,1,2,3],na_values="?")
        raw_data.sort_values(by=['frame','ped'], inplace=True)
        frame = []
        ids = []
        coords = []
        # get_strided_data_clust(raw_data,gt,horizon,1)
        for ped in raw_data.ped.unique(): 
            #for every pedestrian
            ped_data = raw_data[raw_data.ped == ped] #pedestrian data
            num_data_points = ped_data.shape[0] #Number of datapoints for each pedestrian
            src_num = num_data_points - out_gt_num - preds #number of data points for the source
            for i in range (src_num):
                # for every point
                frame.append(ped_data.iloc[i : i + out_gt_num + preds,[0]].values.squeeze()) #data are now in time series manner 
                coords.append(ped_data.iloc[i : i + out_gt_num + preds, 2:4].values)
                ids.append(ped)
            
        frame = np.stack(frame)
        coords = np.stack(coords)
        ids = np.stack(ids)

        peds_speed = np.concatenate((np.zeros((coords.shape[0],1,2)),coords[:,1:,0:2] - coords[:, :-1, 0:2]),1) #calculate the speed
        input_data = np.concatenate((coords,peds_speed),2)

        src_data.append(input_data[: , : out_gt_num])
        trg_data.append(input_data[: , out_gt_num :])
        seq_start_data.append(coords[:,0,:])
        frame_data.append(frame)
        data_num.append(np.array([i]).repeat(coords.shape[0]))
        ped_ids.append(ids)

        
        

    dataset['src'] = np.concatenate(src_data, 0)
    dataset['trg'] = np.concatenate(trg_data, 0)
    dataset['seq_start'] = np.concatenate(seq_start_data, 0)
    dataset['frames'] = np.concatenate(frame_data, 0)
    dataset['dataset'] = np.concatenate(data_num, 0)
    dataset['peds'] = np.concatenate(ped_ids,0)
    mean = dataset['src'].mean((0,1))
    std = dataset['src'].std((0,1))
    return data_TF(dataset, mode, mean, std)
    
class data_TF(Dataset):
    def __init__(self, data, mode, mean, std):
        super().__init__()
        self.data=data
        self.mode=mode

        self.mean= mean
        self.std = std

    def __len__(self):
        return self.data['src'].shape[0]


    def __getitem__(self,index):
        return {'src':torch.Tensor(self.data['src'][index]),
                'trg':torch.Tensor(self.data['trg'][index]),
                'frames':self.data['frames'][index],
                'seq_start':self.data['seq_start'][index],
                'dataset':self.data['dataset'][index],
                'peds': self.data['peds'][index],
                }