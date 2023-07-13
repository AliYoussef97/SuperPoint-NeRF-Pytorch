import importlib
from torch.utils.data import DataLoader

def get_loader(config, task, device="cpu"):
    
    dataset = config["data"]["name"] # Name of dataset script. e.g. Synthetic_data.py
    class_name = config["data"]["class_name"] # Name of class in dataset script. e.g. SyntheticShapes class
    batch_size = config["train"]["batch_size"]

    data_script = importlib.import_module(f"superpoint.data.{dataset}")

    if task == "train":
    
        dataset = {"train":getattr(data_script,class_name)(config["data"], task="train", device=device),
                
                   "validation":getattr(data_script, class_name)(config["data"], task="validation", device=device)}
        
        data_loader = {"train": DataLoader(dataset["train"], 
                                        batch_size=batch_size,
                                        collate_fn=dataset["train"].batch_collator, 
                                        shuffle=True, 
                                        num_workers=0),
                    
                       "validation": DataLoader(dataset["validation"], 
                                                batch_size=batch_size,
                                                collate_fn=dataset["validation"].batch_collator,
                                                shuffle=False,
                                                num_workers=0)}
    if task == "test":
        dataset = {"test":getattr(data_script,class_name)(config["data"], task="test", device=device)}

        data_loader = {"test": DataLoader(dataset["test"],
                                          batch_size=batch_size,
                                          collate_fn=dataset["test"].batch_collator,
                                          shuffle=False,
                                          num_workers=0)}
    
    return data_loader


