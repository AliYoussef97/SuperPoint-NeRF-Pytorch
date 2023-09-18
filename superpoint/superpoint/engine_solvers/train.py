from tqdm import tqdm
import torch
import numpy as np
from pathlib import Path
from itertools import cycle
from superpoint.utils.train_utils import move_to_device
from superpoint.utils.losses import detector_loss, descriptor_loss, descriptor_loss_NeRF
from superpoint.utils.metrics import metrics
from superpoint.settings import CKPT_PATH
from torch.utils.tensorboard import SummaryWriter

def train_val(config, model, 
              train_loader, validation_loader=None,
              mask_loss=False, iteration=0,
              nerf_desc_loss=False, nerf_train=False, 
              device="cpu"):
    
    print(f'\033[92m🚀 Training started for {config["model"]["model_name"].upper()} model on {config["data"]["class_name"]}\033[0m')

    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])

    checkpoint_name = config["ckpt_name"]
    checkpoint_path = Path(CKPT_PATH,checkpoint_name)
    checkpoint_path.mkdir(parents=True,exist_ok=True)   

    writer = SummaryWriter(log_dir = Path(checkpoint_path,"logs"))
 
    max_iterations = config["train"]["num_iters"]    
    iter = iteration

    pbar = tqdm(desc="Training", total=max_iterations, colour="green")
    if iter !=0: pbar.update(iter)
    
    running_loss = []
    
    Train = True

    model.train()

    if nerf_train:
        training_dataloaders = cycle(train_loader)
        if validation_loader is not None:
            validation_dataloaders = cycle(validation_loader)
    
    while Train: 

        if nerf_train:
            train_loader = next(training_dataloaders)    
            if validation_loader is not None:
                validation_loader = next(validation_dataloaders)

        for batch in train_loader:

            batch = move_to_device(batch, device)
            
            output = model(batch["raw"]["image"])

            det_loss = detector_loss(logits= output["detector_output"]["logits"],
                                     kpts_heatmap= batch["raw"]["kpts_heatmap"],
                                     valid_mask= batch["raw"]["valid_mask"],
                                     grid_size= config["model"]["detector_head"]["grid_size"],
                                     include_mask=mask_loss,
                                     device=device)
            writer.add_scalar("iter_loss/Detector loss", det_loss, iter)
            
            loss = det_loss
            

            if config["model"]["model_name"] != "magicpoint":

                warped_output = model(batch["warp"]["image"])

                det_loss_warped = detector_loss(logits=warped_output["detector_output"]["logits"],
                                                kpts_heatmap=batch["warp"]["kpts_heatmap"],
                                                valid_mask=batch["warp"]["valid_mask"],
                                                grid_size=config["model"]["detector_head"]["grid_size"],
                                                include_mask=mask_loss,
                                                device=device)
                writer.add_scalar("iter_loss/Warped Detector loss", det_loss_warped, iter)
                
                if nerf_desc_loss:
                    desc_loss, positive_dist, negative_dist = descriptor_loss_NeRF(config=config["model"],
                                                                                   data=batch,
                                                                                   descriptors=output["descriptor_output"]["desc_raw"],
                                                                                   warped_descriptors=warped_output["descriptor_output"]["desc_raw"],
                                                                                   valid_mask=batch["warp"]["valid_mask"],
                                                                                   include_mask=mask_loss,
                                                                                   device=device)

                else:    
                    desc_loss, positive_dist, negative_dist = descriptor_loss(config=config["model"],
                                                                              descriptors=output["descriptor_output"]["desc_raw"],
                                                                              warped_descriptors=warped_output["descriptor_output"]["desc_raw"],
                                                                              homographies=batch["homography"],
                                                                              valid_mask=batch["warp"]["valid_mask"],
                                                                              include_mask=mask_loss,
                                                                              device=device)
                
                writer.add_scalar("iter_loss/Descriptor loss", desc_loss, iter)
                writer.add_scalar("margins/Positive Distribution", positive_dist, iter)
                writer.add_scalar("margins/Negative Distribution", negative_dist, iter)
                
                loss += (det_loss_warped + desc_loss)

            running_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            iter += 1
            pbar.update(1)
            
            if iter % config["save_or_validation_interval"] == 0:

                running_loss = np.mean(running_loss)           
                writer.add_scalar("running_loss/Training loss", running_loss, iter)

                if validation_loader is not None:
                    model.eval()
                
                    running_val_loss, precision, recall = validate(config, model, validation_loader, mask_loss, nerf_desc_loss, device=device)

                    model.train()
                        
                    writer.add_scalar("running_loss/Validation loss", running_val_loss, iter)
                    writer.add_scalar("metrics/Precision", precision, iter)
                    writer.add_scalar("metrics/Recall", recall, iter)
                    
                    tqdm.write('Iteration: {}, Running Training loss: {:.4f}, Running Validation loss: {:.4f}, Precision: {:.4f}, Recall: {:.4f}'
                               .format(iter, running_loss, running_val_loss, precision, recall))
                
                else:
                    tqdm.write('Iteration: {}, Running Training loss: {:.4f}'
                               .format(iter, running_loss))

                torch.save({"iteration":iter,
                            "model_state_dict":model.state_dict()},
                            Path(checkpoint_path,f'{checkpoint_name}_{iter}.pth'))
                
                running_loss = []
                

            if iter == max_iterations:

                torch.save({"iteration":iter,
                            "model_state_dict":model.state_dict()},
                            Path(checkpoint_path,f'{checkpoint_name}_{iter}.pth'))
                Train = False
                writer.flush()
                writer.close()
                pbar.close()
                print(f'\033[92m✅ {config["model"]["model_name"].upper()} Training finished\033[0m')
                break

@torch.no_grad()         
def validate(config, model, validation_loader, mask_loss, nerf_desc_loss , device= "cpu"):
    
    running_val_loss = []
    precision = []
    recall = []

    for val_batch in tqdm(validation_loader, desc="Validation",colour="blue"):

        val_batch = move_to_device(val_batch, device)
        
        val_output = model(val_batch["raw"]["image"])
        
        val_det_loss = detector_loss(logits=val_output["detector_output"]["logits"],
                                     kpts_heatmap=val_batch["raw"]["kpts_heatmap"],
                                     valid_mask=val_batch["raw"]["valid_mask"],
                                     grid_size=config["model"]["detector_head"]["grid_size"],
                                     include_mask=mask_loss,
                                     device=device)
        
        val_loss = val_det_loss
        
        if config["model"]["model_name"] != "magicpoint":

            val_warped_output = model(val_batch["warp"]["image"])

            val_det_loss_warped = detector_loss(logits=val_warped_output["detector_output"]["logits"],
                                                kpts_heatmap=val_batch["warp"]["kpts_heatmap"],
                                                valid_mask=val_batch["warp"]["valid_mask"],
                                                grid_size=config["model"]["detector_head"]["grid_size"],
                                                include_mask=mask_loss,
                                                device=device)
            
            if nerf_desc_loss:
                val_desc_loss, _, _ = descriptor_loss_NeRF(config=config["model"],
                                                     data=val_batch,
                                                     descriptors=val_output["descriptor_output"]["desc_raw"],
                                                     warped_descriptors=val_warped_output["descriptor_output"]["desc_raw"],
                                                     valid_mask=val_batch["warp"]["valid_mask"],
                                                     include_mask=mask_loss,
                                                     device=device)


            else:
                val_desc_loss, _, _ = descriptor_loss(config=config["model"],
                                                descriptors=val_output["descriptor_output"]["desc_raw"],
                                                warped_descriptors=val_warped_output["descriptor_output"]["desc_raw"],
                                                homographies=val_batch["homography"],
                                                valid_mask=val_batch["warp"]["valid_mask"],
                                                include_mask=mask_loss,
                                                device=device)
            
            val_loss += (val_det_loss_warped + val_desc_loss)
        
        running_val_loss.append(val_loss.item())

        metric = metrics(val_output["detector_output"],val_batch["raw"])
        
        precision.append(metric["precision"])
        recall.append(metric["recall"])
    
    running_val_loss = np.mean(running_val_loss)
    precision = np.mean(precision)
    recall = np.mean(recall)
        
    return running_val_loss, precision, recall