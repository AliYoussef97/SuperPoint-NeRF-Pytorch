from tqdm import tqdm
import torch
from pathlib import Path
from superpoint.utils.losses import detector_loss, descriptor_loss
from superpoint.utils.metrics import metrics
from superpoint.settings import EXPER_PATH
from torch.utils.tensorboard import SummaryWriter

def train_val(config, model, train_loader, validation_loader, device="cpu"):
    print(f'\033[92m🚀 Training started for {config["model"]["model_name"].upper()} model on {config["data"]["class_name"]}\033[0m')

    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"],betas=config["train"]["betas"])

    
    checkpoint_name = config["ckpt_name"]
    checkpoint_path = Path(EXPER_PATH,checkpoint_name)
    checkpoint_path.mkdir(parents=True,exist_ok=True)   

    writer = SummaryWriter(log_dir = f'{checkpoint_path}\logs')
 
    max_iterations = config["train"]["num_iters"]    
    Train = True
    iter = config["iteration"] if config["iteration"] else 0
    running_loss = []
    
    pbar = tqdm(desc="Training", total=max_iterations, colour="green")
    if iter>0: pbar.update(iter)
    
    while Train:
        model.train()
        
        for batch in train_loader:
            
            if config["model"]["model_name"] == "magicpoint" and config["data"]["class_name"] == "COCO":
                print("Loading COCO data")
                batch["raw"] = batch["warp"]

            output = model(batch["raw"]["image"])

            det_loss = detector_loss(output["detector_output"]["logits"],
                                     batch["raw"]["kpts_heatmap"],
                                     batch["raw"]["valid_mask"],
                                     config["model"]["detector_head"]["grid_size"],
                                     device=device)
            
            loss = det_loss
            
            if config["model"]["model_name"] != "magicpoint":
                print("Superpoint model")
                warped_output = model(batch["warp"]["image"])

                det_loss_warped = detector_loss(warped_output["detector_output"]["logits"],
                                                batch["warp"]["kpts_heatmap"],
                                                batch["warp"]["valid_mask"],
                                                config["model"]["detector_head"]["grid_size"],
                                                device=device)
                
                desc_loss = descriptor_loss(config["model"],
                                            output["descriptor_output"]["desc_raw"],
                                            warped_output["descriptor_output"]["desc_raw"],
                                            batch["homography"],
                                            batch["warp"]["valid_mask"],
                                            device=device)
                
                loss += det_loss_warped + desc_loss

            running_loss.append(loss.item())

            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            iter += 1
            pbar.update(1)
            
            if iter % config["validation_interval"] == 0:

                running_loss = sum(running_loss)/float(len(running_loss))            

                running_val_loss, precision, recall = validate(config, model, validation_loader, device=device)

                model.train()

                torch.save({"iteration":iter,
                            "model_state_dict":model.state_dict()},
                            f'{checkpoint_path}\{checkpoint_name}.pth')
                
                tqdm.write('Iteration: {}, Running Training loss: {:.3f}, Running Validation loss: {:.3f}, Precision: {:.3f}, Recall: {:.3f}'
                           .format(iter, running_loss, running_val_loss, precision, recall))

                writer.add_scalar("Training loss", running_loss, iter)
                writer.add_scalar("Validation loss", running_val_loss, iter)
                writer.add_scalar("Precision", precision, iter)
                writer.add_scalar("Recall", recall, iter)
                
                running_loss = []

            if iter == max_iterations:

                torch.save({"iteration":iter,
                            "model_state_dict":model.state_dict()},
                            f'{checkpoint_path}\{checkpoint_name}.pth')
                Train = False
                writer.flush()
                writer.close()
                pbar.close()
                print(f'\033[92m✅ {config["model"]["model_name"].upper()} Training finished\033[0m')
                break

            
def validate(config, model, validation_loader, device= "cpu"):
    
    model.eval()

    running_val_loss = []
    precision = []
    recall = []

    with torch.no_grad():
        for val_batch in tqdm(validation_loader, desc="Validation",colour="blue"):
            
            if config["model"]["model_name"] == "magicpoint" and config["data"]["class_name"] == "COCO":
                print("VAL Loading COCO data")
                val_batch["raw"] = val_batch["warp"]


            val_output = model(val_batch["raw"]["image"])
            
            val_det_loss = detector_loss(val_output["detector_output"]["logits"],
                                         val_batch["raw"]["kpts_heatmap"],
                                         val_batch["raw"]["valid_mask"],
                                         config["model"]["detector_head"]["grid_size"],
                                         device=device)
            
            val_loss = val_det_loss
            
            if config["model"]["model_name"] != "magicpoint":
                print(" VAL Superpoint model")

                val_warped_output = model(val_batch["warped"]["image"])

                val_det_loss_warped = detector_loss(val_warped_output["detector_output"]["logits"],
                                                    val_batch["warp"]["kpts_heatmap"],
                                                    val_batch["warp"]["valid_mask"],
                                                    config["model"]["detector_head"]["grid_size"],
                                                    device=device)
                
                val_desc_loss = descriptor_loss(config["model"],
                                                val_output["descriptor_output"]["desc_raw"],
                                                val_warped_output["descriptor_output"]["desc_raw"],
                                                val_batch["homography"],
                                                val_batch["warp"]["valid_mask"],
                                                device=device)
                
                val_loss += val_det_loss_warped + val_desc_loss
            
            running_val_loss.append(val_loss.item())

            metric = metrics(val_output["detector_output"],val_batch["raw"])
            
            precision.append(metric["precision"])
            recall.append(metric["recall"])
        
        running_val_loss = sum(running_val_loss)/float(len(running_val_loss))
        precision = sum(precision)/float(len(precision))
        recall = sum(recall)/float(len(recall))
        
    return running_val_loss, precision, recall