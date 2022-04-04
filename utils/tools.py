import os
import datetime
import json
import importlib
from datetime import datetime
import torch

def make_folders(exp_name):

    model_path = f"./experiments/{exp_name}"
    log_file = f"{model_path}/log_{str(datetime.now())[:-7]}.txt"

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    return model_path, log_file

def map_config(config):

    if "resize" in config.keys():
        # Change str to bool.
        config["resize"] = json.loads(config["resize"].lower())

    # Now convert model, criterion, optimzier, scheduler
    model = convert_to_obj(config, "model")
    model = model(n_classes=config["n_classes"])
    # model.load_state_dict(torch.load("./experiments/FERPLUS_RED_with_PDLS_WITH_STN_AND_HARD_DA/best.pth")["model"])
    
    criterion = convert_to_obj(config, "criterion")
    criterion = criterion()

    optimizer = convert_to_obj(config, "optimizer")
    optimizer = optimizer(model.parameters(), lr=config["lr"])

    scheduler = convert_to_obj(config, "scheduler")
    scheduler = scheduler(optimizer, step_size = 15, gamma=0.25)

    config["model"] = model
    config["criterion"] = criterion
    config["optimizer"] = optimizer
    config["scheduler"] = scheduler

    return config

def convert_to_obj(config, target):
    mod_list = config[target].split(".")
    module = importlib.import_module('.'.join(mod_list[:-1]))
    return getattr(module, mod_list[-1])
