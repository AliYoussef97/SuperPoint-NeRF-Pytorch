import importlib


def get_model(config,device="cpu"):

    script = config["script"] # Name of model script. e.g. SuperPoint.py
    class_name = config["class_name"] # Name of class in model script. e.g. SuperPoint class

    model_script = importlib.import_module(f"superpoint.models.{script}")
    model = getattr(model_script, class_name)(config)
        
    return model.to(device)