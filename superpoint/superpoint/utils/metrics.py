import torch

def metrics(outputs, inputs):
   
    pred = inputs["valid_mask"] * outputs["pred_pts"]
    labels = inputs["kpts_heatmap"]

    precision = torch.sum(pred * labels) / torch.sum(pred+10e-6)
    recall = torch.sum(pred * labels) / torch.sum(labels+10e-6)
  
    return {"precision":precision.item(), "recall":recall.item()}