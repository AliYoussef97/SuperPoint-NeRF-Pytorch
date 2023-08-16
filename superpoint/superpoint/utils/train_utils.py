import torch


def move_to_device(obj, device):
  if torch.is_tensor(obj):
    return obj.to(device)
  elif isinstance(obj, dict):
    out = {}
    for k, v in obj.items():
      out[k] = move_to_device(v, device)
    return out
  elif isinstance(obj, list):
    out = []
    for v in obj:
      out.append(move_to_device(v, device))
    return out
  else:
    return obj