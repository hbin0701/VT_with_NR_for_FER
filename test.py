import json
import argparse
from utils.dataloaders import create_data_loaders
from utils.train_model import train_model
from utils.tools import make_folders, map_config, convert_to_obj
import os
import shutil
import torch
import numpy as np
import random

parser = argparse.ArgumentParser(description="Running our model...")
parser.add_argument('-c', '--config', help='path to config file')

def main():
    args = parser.parse_args()

    # SEED INITALIZATION
    random_seed = 1111
    
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
        
    with open(args.config, "r") as f:
        
        configs = map_config(json.load(f))
       
        # Make exp directory
        exp_name = configs['exp_name']
        
        model_path, log_file = make_folders(exp_name)
        
        shutil.copyfile(args.config, os.path.join(model_path, "config.json"))
        
        
        if "aligned" not in configs.keys():
            configs['aligned'] = None
            
        
        print("Loading DataLoaders...")
        
        train_loader, test_loader = create_data_loaders(configs['batch_size'], 
                                                        configs['resize'], 
                                                        configs['data_dir'], 
                                                        configs['dataset'], 
                                                        configs['aligned'])
        
        print("Starting Training...")
        
        train_model(configs["model"], 
                    log_file, 
                    train_loader, 
                    test_loader, 
                    configs["criterion"],
                    configs["optimizer"], 
                    configs["scheduler"], 
                    configs["num_epochs"])

 
        

if __name__ == '__main__':
    main()