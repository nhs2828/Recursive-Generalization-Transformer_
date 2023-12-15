from ignite.metrics import PSNR, SSIM
import wandb
from ignite.engine import Engine
from data_preparation import *
import distutils.version
import torchvision
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
#from tqdm import tqdm
from pathlib import Path
from model import *

transforms = LRImgTransforms(64)
target_transforms = HRImgTransforms(128)

path = '../../DIV2K_train_HR'
path_val = '../../DIV2K_valid_HR'

train_ds = CustomSuperResolutionDataset(path,\
                                        transform=transforms.transforms,\
                                        target_transform=target_transforms.transforms)

val_ds = CustomSuperResolutionDataset(path_val,\
                                        transform=transforms.transforms,\
                                        target_transform=target_transforms.transforms)

train_dl = DataLoader(train_ds, 32, shuffle=True)
val_dl = DataLoader(val_ds, 32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class State :
    def __init__ (self, model, optim, scheduler):
        self.model = model
        self.optim = optim
        self.scheduler = scheduler
        self.iteration = 0
        
savepath = Path("RGT.plt")

iterations = 500000
h_train = 4
h_test = 16
LR = 2e-4
milestones = [250000, 400000, 450000, 475000]
if savepath.is_file():
    with savepath.open("rb") as fp :
        state = torch.load(fp)
else:
    # model
    model = RGT(C = 180,\
            dim_in = 3,\
            N1 = 8,\
            N2 = 6,\
            cr = 0.5,\
            s_r = 4,\
            hidden_ratio = 2,\
            upscale_factor = 2
           )
    model = model.to(device)
    # optim
    optim = torch.optim.Adam(params = model.parameters(), lr = LR, betas=(0.9, 0.99))
    # scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=milestones, gamma=0.5)
    state = State(model, optim, scheduler)

# loss
loss_fn = torch.nn.L1Loss()
# metrics
def process_function(engine, batch):
    y_pred, y = batch[0], batch[1]
    return y_pred, y

default_evaluator = Engine(process_function)

psnr = PSNR(data_range=1.0, output_transform=get_y_channel)
psnr.attach(default_evaluator, 'psnr')
ssim = SSIM(data_range=1.0)
ssim.attach(default_evaluator, 'ssim')

wandb.init(project="RGT", sync_tensorboard=True)

# train
layout = {
    "Toto": {
        "loss": ["Multiline", ["loss/train", "loss/val"]],
        "psnr": ["Multiline", ["psnr/train", "psnr/val"]],
        "ssim": ["Multiline", ["ssim/train", "ssim/val"]]
    },
}

writer = SummaryWriter()
writer.add_custom_scalars(layout)

while(state.iteration < iterations):
    train_loss = 0.
    psnr_train = 0.
    ssim_train = 0.
    state.model.train()
    for x, x_sr in train_dl:
        # data
        x, x_sr = x.to(device), x_sr.to(device)
        # super resolution
        pred_sr = state.model(x, h_train)
        
        # loss and update
        state.optim.zero_grad()
        l = loss_fn(pred_sr, x_sr)
        l.backward()
        state.optim.step()
        state.iteration += 1
        state.scheduler.step()
        train_loss += l.item()
        
        state_metrics = default_evaluator.run([[pred_sr.cpu(), x_sr.cpu()]])
        psnr_train += state_metrics.metrics['psnr']
        ssim_train += state_metrics.metrics['ssim']
        
    writer.add_scalar("loss/train", train_loss/len(train_dl), state.iteration)
    writer.add_scalar("psnr/train", psnr_train/len(train_dl), state.iteration)
    writer.add_scalar("ssim/train", ssim_train/len(train_dl), state.iteration)
        
    with savepath.open("wb") as fp :
        torch.save(state, fp)
    
    with torch.no_grad():
        state.model.eval()
        val_loss = 0.
        psnr_val = 0.
        ssim_val = 0.
        for x_val,x_val_sr in val_dl:
            # data
            x_val, x_val_sr = x_val.to(device), x_val_sr.to(device)
            # super resolution
            pred_val_sr = state.model(x_val, h_test)
            # loss
            l_val = loss_fn(pred_val_sr, x_val_sr)
            val_loss += l_val.item()
            
            state_metrics = default_evaluator.run([[pred_sr.cpu(), x_sr.cpu()]])
            psnr_val += state_metrics.metrics['psnr']
            ssim_val += state_metrics.metrics['ssim']
        writer.add_scalar("loss/val", val_loss/len(val_dl), state.iteration)
        writer.add_scalar("psnr/val", psnr_val/len(val_dl), state.iteration)
        writer.add_scalar("ssim/val", ssim_val/len(val_dl), state.iteration)
        # visualization
        grid_img = torchvision.utils.make_grid(pred_val_sr[:10], nrow=5)
        grid_img_HR = torchvision.utils.make_grid(x_val_sr[:10], nrow=5)
        writer.add_images('SR', grid_img, state.iteration, dataformats='CHW')
        writer.add_images('True_HR', grid_img_HR, state.iteration, dataformats='CHW')
writer.close()
wandb.finish()
