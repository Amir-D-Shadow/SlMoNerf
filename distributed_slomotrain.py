import os, random, datetime, sys
from tkinter import TRUE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import imageio
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms
import torch.distributed as dist
from einops import rearrange
from PIL import Image as Image

# utlities
from nerfmm.utils.pos_enc import encode_position
from nerfmm.utils.volume_op import volume_rendering, volume_sampling_ndc
from nerfmm.utils.comp_ray_dir import comp_ray_dir_cam_fxfy
from nerfmm.utils.lie_group_helper import convert3x4_4x4
from nerfmm.utils.pose_utils import create_spiral_poses
from nerfmm.utils.training_utils import mse2psnr

from loss_utils import *
from model import *
from helper import flow_inp_frame, render_inp_frame

import argparse

def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()
def de_parallel(model):
    return model.module if hasattr(model, 'module') else model

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0,
                        help="the id of the current running process")
    return parser

#setting
parser = config_parser()
args = parser.parse_args()

cwd = sys.path[0]
base_dir = f'{cwd}/logs'
expname = "room3_testing"
os.makedirs(os.path.join(base_dir, expname), exist_ok=True)

scene_name = "room3"
img_dir = f"{cwd}/data/{scene_name}"
model_weight_dir = f"{cwd}/model_weights/{scene_name}"
ckpt_path = f"{cwd}/ckpt/SuperSloMo.ckpt"

load_model = True
N_EPOCH = 1000  # set to 1000 to get slightly better results. we use 10K epoch in our paper.
EVAL_INTERVAL = 60  # render an image to visualise for every this interval.
SAVE_WEIGHTS_INTERVAL =  20
SAVE_IMAGE_INTERVAL = 200

device = torch.device(f"cuda:{args.local_rank}")
torch.distributed.init_process_group(backend="nccl", init_method="env://")
synchronize()


SSIM_loss = SSIM(size_average=True)
# SSIM_loss = nn.DataParallel(SSIM_loss,device_ids=device_ids)
SSIM_loss.to(device)

#load data
def load_imgs(image_dir):

    mean = [0.429, 0.431, 0.397]
    std  = [1, 1, 1]
    normalize = transforms.Normalize(mean=mean,std=std)

    transform = transforms.Compose([transforms.ToTensor(), normalize])
   
    img_names = np.array(sorted(os.listdir(image_dir)))  # all image names
    img_paths = [os.path.join(image_dir, n) for n in img_names]
    N_imgs = len(img_paths)

    img_list = []
    for p in img_paths:
        img = imageio.imread(p)[:, :, :3]  # (H, W, 3) np.uint8
        img = Image.fromarray(img).resize((512,288),Image.BILINEAR) #reshape (640,360)
        img = transform(img)  # (3,H, W) 
        img_list.append(img)

    img_list = torch.stack(img_list,dim=0).float() #(N, 3, H, W) torch.float32

    H, W = img_list.shape[2], img_list.shape[3]
    
    results = {
        'imgs': img_list,  #(N, 3, H, W) torch.float32
        'img_names': img_names,  # (N, )
        'N_imgs': N_imgs,
        'H': H,
        'W': W,
    }
    return results

#load image data
def load_img_data(image_dir):

    folders = os.listdir(image_dir)
    folders.sort(key = lambda x: int(x))
    TSteps = len(folders)

    results = {}
    images_data = {}
    
    for i in range(TSteps):

        path = f"{image_dir}/{folders[i]}/images"
        
        image_info = load_imgs(path) 
        imgs = image_info['imgs']      #(N, 3, H, W) torch.float32

        #save imgs pack
        images_data[i] = imgs
        
        N_IMGS = image_info['N_imgs']
        H = image_info['H']
        W = image_info['W']

    results = {
        'images_data': images_data,  # dict
        'TSteps':TSteps,
        'N_IMGS': N_IMGS,
        'H': H,
        'W': W,
    }

    return results

image_info = load_img_data(img_dir)

image_data = image_info['images_data']  #(N, 3, H, W) torch.float32
N_IMGS = image_info['N_IMGS'] 
H = image_info['H']
W = image_info['W']
TSteps = image_info["TSteps"]

print('Loaded {0} imgs, resolution {1} x {2}'.format(N_IMGS*TSteps, H, W))

#learn focal
class LearnFocal(nn.Module):
    def __init__(self, H, W, req_grad):
        super(LearnFocal, self).__init__()
        self.H = H
        self.W = W
        self.fx = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
        self.fy = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )

    def forward(self):
        # order = 2, check our supplementary.
        fxfy = torch.stack([self.fx**2 * self.W, self.fy**2 * self.W])
        return fxfy

#Learn Pose
def vec2skew(v):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    zero = torch.zeros(1, dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([ zero,    -v[2:3],   v[1:2]])  # (3, 1)
    skew_v1 = torch.cat([ v[2:3],   zero,    -v[0:1]])
    skew_v2 = torch.cat([-v[1:2],   v[0:1],   zero])
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
    return skew_v  # (3, 3)


def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    skew_r = vec2skew(r)  # (3, 3)
    norm_r = r.norm() + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    return R


def make_c2w(r, t):
    """
    :param r:  (3, ) axis-angle             torch tensor
    :param t:  (3, ) translation vector     torch tensor
    :return:   (4, 4)
    """
    R = Exp(r)  # (3, 3)
    c2w = torch.cat([R, t.unsqueeze(1)], dim=1)  # (3, 4)
    c2w = convert3x4_4x4(c2w)  # (4, 4)
    return c2w


class LearnPose(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t):
        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

    def forward(self, cam_id):
        r = self.r[cam_id]  # (3, ) axis-angle
        t = self.t[cam_id]  # (3, )
        c2w = make_c2w(r, t)  # (4, 4)
        return c2w

#define learnable time
class LearnTime(nn.Module):

   def __init__(self,in_feat,mid_feat,out_feat):

      super(LearnTime,self).__init__()

      self.TimeLayer1 = nn.Linear(in_features=in_feat,out_features=mid_feat)
      self.TimeLayer2 = nn.Linear(in_features=mid_feat,out_features=out_feat)
      self.act_fn1 = nn.ReLU()

   def forward(self,x):

      """
      x (time tensor)  
      """

      timeStep = self.TimeLayer1(x)
      timeStep = self.act_fn1(timeStep)

      final_output = self.TimeLayer2(timeStep)

      return final_output

#Nerf
class Nerf(nn.Module):
    def __init__(self, tpos_in_dims, dir_in_dims, D):
        """
        :param pos_in_dims: scalar, number of channels of encoded positions
        :param dir_in_dims: scalar, number of channels of encoded directions
        :param D:           scalar, number of hidden dimensions
        """
        super(Nerf, self).__init__()

        self.embedding = nn.Linear(tpos_in_dims,D)

        self.layers0 = nn.Sequential(
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )

        self.layers1 = nn.Sequential(
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )

        #density
        self.fc_density = nn.Linear(D, 1)

        #color
        self.fc_feature = nn.Linear(D, D)
        self.rgb_layers = nn.Sequential(nn.Linear(D + dir_in_dims, D//2), nn.ReLU())
        self.fc_rgb = nn.Linear(D//2, 3)

        #bias
        self.fc_density.bias.data = torch.tensor([0.1]).float()
        self.fc_rgb.bias.data = torch.tensor([0.02, 0.02, 0.02]).float()

    def forward(self, tpos_enc, dir_enc):
        """
        :param time_enc: (H, W, N_sample, time_in_dims) encoded time
        :param pos_enc: (H, W, N_sample, pos_in_dims) encoded positions
        :param dir_enc: (H, W, N_sample, dir_in_dims) encoded directions
        :return: rgb_density (H, W, N_sample, 4)
        """
        #cat pos and time
        pos_time_enc = self.embedding(tpos_enc) # (H, W, N_sample, D)

        #forward
        x = self.layers0(pos_time_enc)  # (H, W, N_sample, D)
        y = x + pos_time_enc  # (H, W, N_sample, D)
        y = self.layers1(y)   # (H, W, N_sample, D)

        #density
        density = self.fc_density(y)  # (H, W, N_sample, 1)

        #Color
        feat = self.fc_feature(y)  # (H, W, N_sample, D)
        feat = torch.cat([feat, dir_enc], dim=3)  # (H, W, N_sample, D+dir_in_dims)
        feat = self.rgb_layers(feat)  # (H, W, N_sample, D/2)
        rgb = self.fc_rgb(feat)  # (H, W, N_sample, 3)

        #output
        rgb_den = torch.cat([rgb, density], dim=3)  # (H, W, N_sample, 4)

        return rgb_den

#Set ray parameters
class RayParameters():
    def __init__(self):
      self.NEAR, self.FAR = 0.0, 1.0  # ndc near far
      self.N_SAMPLE = 128  # samples per ray
      self.POS_ENC_FREQ = 10  # positional encoding freq for location
      self.DIR_ENC_FREQ = 4   # positional encoding freq for direction
      self.num_sample_steps = TSteps - 1
      self.num_inp_frame = 4

ray_params = RayParameters()

#Define training function***
def model_render_image(time_pose_net,T_momen,c2w, rays_cam, t_vals, ray_params, H, W, fxfy, nerf_model,perturb_t, sigma_noise_std):

    """
    :param time_pose_net  model that learn time
    :param T_momen        current time step
    :param c2w:         (4, 4)                  pose to transform ray direction from cam to world.
    :param rays_cam:    (someH, someW, 3)       ray directions in camera coordinate, can be random selected
                                                rows and cols, or some full rows, or an entire image.
    :param t_vals:      (N_samples)             sample depth along a ray.
    :param perturb_t:   True/False              perturb t values.
    :param sigma_noise_std: float               add noise to raw density predictions (sigma).
    :return:            (someH, someW, 3)       volume rendered images for the input rays.
    """

    # (H, W, N_sample, 3), (H, W, 3), (H, W, N_sam)
    sample_pos, _, ray_dir_world, t_vals_noisy = volume_sampling_ndc(c2w, rays_cam, t_vals, ray_params.NEAR,ray_params.FAR, H, W, fxfy, perturb_t)

    #bulid Time tensor (H,W,N_sample,out_feat)
    timeTensor = torch.full_like(ray_dir_world,fill_value=T_momen,dtype=torch.float32,device=ray_dir_world.device) #(H,W,3)
    timeTensor = time_pose_net(timeTensor)                                    #(H,W,3)
    timeTensor = timeTensor.unsqueeze(2).expand(-1,-1,ray_params.N_SAMPLE,-1) #(H,W,N_sample,3)
    
    # encode position: (H, W, N_sample, (2L+1)*C = 126)
    tpos = torch.cat([sample_pos,timeTensor],dim=-1) #(H,W,N_sample,6)
    tpos_enc = encode_position(tpos, levels=ray_params.POS_ENC_FREQ, inc_input=True)

    # encode direction: (H, W, N_sample, (2L+1)*C = 27)
    ray_dir_world = F.normalize(ray_dir_world, p=2, dim=2)  # (H, W, 3)
    dir_enc = encode_position(ray_dir_world, levels=ray_params.DIR_ENC_FREQ, inc_input=True)  # (H, W, 27)
    dir_enc = dir_enc.unsqueeze(2).expand(-1, -1, ray_params.N_SAMPLE, -1)  # (H, W, N_sample, 27)

    # inference rgb and density using position and direction encoding.
    rgb_density = nerf_model(tpos_enc, dir_enc)  # (H, W, N_sample, 4)

    render_result = volume_rendering(rgb_density, t_vals_noisy, sigma_noise_std, rgb_act_fn=torch.sigmoid)
    rgb_rendered = render_result['rgb']  # (H, W, 3)
    depth_map = render_result['depth_map']  # (H, W)

    result = {
        'rgb': rgb_rendered,  # (H, W, 3)
        'depth_map': depth_map,  # (H, W)
    }

    return result

#***
def train_one_epoch(images_data, H, W, ray_params, opt_nerf, opt_focal,opt_pose,opt_time, nerf_model, focal_net, pose_param_net,time_pose_net,flowComp,ArbTimeFlowIntrp,flowBackWarp):

    """
    images_data: {0:(N,H,W,C),1:(N,H,W,C),2:(N,H,W,C),...}   dictionary that stores images for each time step with N camera pose each
    """

    nerf_model.train()
    focal_net.train()
    pose_param_net.train()
    time_pose_net.train()

    #set up
    mean = [0.429, 0.431, 0.397]
    std  = [1, 1, 1]

    negmean = [x * -1 for x in mean]
    revNormalize = transforms.Normalize(mean=negmean, std=std)
    
    TP = transforms.Compose([revNormalize,transforms.ToPILImage()])

    t_vals = torch.linspace(ray_params.NEAR, ray_params.FAR, ray_params.N_SAMPLE, device=device)  # (N_sample,) sample position

    total_loss_epoch = []
    L2_loss_epoch = []
    ssim_loss_epoch = []
    
    #shuffle the time steps
    time_list = [ i for i in range(1, TSteps-1)]
    random.shuffle(time_list)

    timeSteps = [time_list[i] for i in range(ray_params.num_sample_steps-1)]
    random.shuffle(timeSteps)

    for t in timeSteps:
        imgs0 = images_data[t].to(device) 
        imgs1 = images_data[t+1].to(device)  #(N, 3, H, W)
        
        # shuffle the training imgs
        ids = np.arange(N_IMGS)
        np.random.shuffle(ids)

        for intermediateInpidx in range(ray_params.num_inp_frame):

            t_idx = (t + intermediateInpidx / ray_params.num_inp_frame) / TSteps
            # TSteps: number of frames 

            #render inp frame
            if intermediateInpidx == 0:

                imgs = [] 
                for i in range(N_IMGS):
                    imgs.append(TP(imgs0[i]))

                imgs = torch.from_numpy( np.stack(imgs,axis=0) ).float() #(N,H,W,3)

            else:

                imgs = render_inp_frame(frame0 = imgs0,                                         #(N,H,W,3) 
                                        frame1 = imgs1,
                                        intermediateIndex = intermediateInpidx,
                                        num_inp_frame = ray_params.num_inp_frame,
                                        flowComp = flowComp,
                                        ArbTimeFlowIntrp = ArbTimeFlowIntrp,
                                        flowBackWarp = flowBackWarp,
                                        ) 
        for i in ids:

            fxfy = de_parallel(focal_net)()
            
            # KEY 1: compute ray directions using estimated intrinsics online.
            ray_dir_cam = comp_ray_dir_cam_fxfy(H, W, fxfy[0], fxfy[1])
            img = imgs[i].to(device)            # (H, W, 3)

            c2w = pose_param_net(i)  # (4, 4)
            dH, dW = 50, 50
            # sample 32x32 pixel on an image and their rays for training.
            r_id = torch.randperm(H, device=device)[:dH]  # (N_select_rows)
            c_id = torch.randperm(W, device=device)[:dW]  # (N_select_cols)
            ray_selected_cam = ray_dir_cam[r_id][:, c_id]  # (N_select_rows, N_select_cols, 3)
            img_selected = img[r_id][:, c_id]  # (N_select_rows, N_select_cols, 3



            # render an image using selected rays, pose, sample intervals, and the network

            render_result = model_render_image(time_pose_net,t_idx,c2w, ray_selected_cam, t_vals, ray_params,H, W, fxfy, nerf_model, perturb_t=True, sigma_noise_std=0.0)
            
            
            rgb_rendered = render_result['rgb'] * 255.0  # (N_select_rows, N_select_cols, 3)
            depth_rendered = render_result['depth_map'] * 200.0

            # #l1 loss
            # rgb_l1_loss = F.l1_loss(rgb_rendered,img_selected)

            # #ssim
            # ssim_syn = rearrange( rgb_rendered.unsqueeze(0), "b h w c -> b c h w") # (1,N_select_rows, N_select_cols, 3) -> (1,3,N_select_rows, N_select_cols)
            # ssim_tgt = rearrange( img_selected.unsqueeze(0), "b h w c -> b c h w") # (1,N_select_rows, N_select_cols, 3) -> (1,3,N_select_rows, N_select_cols)
            # rgb_ssim_loss =  1 - SSIM_loss(ssim_syn,ssim_tgt)
            
            # #edge_aware_loss
            # disp =  torch.reciprocal(depth_rendered+1e-7) # (N_select_rows, N_select_cols)
            # EAL_loss = edge_aware_loss(img_selected,disp)

            #L2_loss
            L2_loss = F.mse_loss(rgb_rendered/255.0, img_selected/255.0)

            #total loss
            total_loss = L2_loss  #+ 0.3*rgb_l1_loss + 0.5*rgb_ssim_loss + 0.01*EAL_loss

            #L2_loss.backward()
            total_loss.backward()
            opt_nerf.step()
            opt_focal.step()
            opt_pose.step()
            opt_time.step()
            
            opt_nerf.zero_grad()
            opt_focal.zero_grad()
            opt_pose.zero_grad()
            opt_time.zero_grad()

            with torch.no_grad():

                L2_loss_epoch.append(L2_loss.clone().detach())




    L2_loss_epoch_mean = torch.stack(L2_loss_epoch).mean().item()


    #[L2_loss_epoch_mean,0,total_loss_epoch_mean]
    return L2_loss_epoch_mean

#Define evaluation function***
def render_novel_view(T_momen,c2w, H, W, fxfy, ray_params, nerf_model,time_pose_net):
    nerf_model.eval()

    ray_dir_cam = comp_ray_dir_cam_fxfy(H, W, fxfy[0], fxfy[1])
    t_vals = torch.linspace(ray_params.NEAR, ray_params.FAR, ray_params.N_SAMPLE, device=device)  # (N_sample,) sample position

    c2w = c2w.to(device)  # (4, 4)

    # split an image to rows when the input image resolution is high
    rays_dir_cam_split_rows = ray_dir_cam.split(10, dim=0)  # input 10 rows each time
    rendered_img = []
    rendered_depth = []
    
    for rays_dir_rows in rays_dir_cam_split_rows:

        
        render_result = model_render_image(time_pose_net,T_momen,c2w, rays_dir_rows, t_vals, ray_params,
                                           H, W, fxfy, nerf_model,
                                           perturb_t=False, sigma_noise_std=0.0)
                                           
        rgb_rendered_rows = render_result['rgb']  # (num_rows_eval_img, W, 3)
        depth_map = render_result['depth_map']  # (num_rows_eval_img, W)

        rendered_img.append(rgb_rendered_rows)
        rendered_depth.append(depth_map)

    # combine rows to an image
    rendered_img = torch.cat(rendered_img, dim=0)  # (H, W, 3)
    rendered_depth = torch.cat(rendered_depth, dim=0)  # (H, W)
    return rendered_img, rendered_depth


#training
# Initialize SlMo model
dict1 = torch.load(ckpt_path, map_location='cpu')

flowComp = UNet(6, 4)
flowComp.load_state_dict(dict1['state_dictFC'])
for param in flowComp.parameters():
    param.requires_grad = False
# flowComp = nn.DataParallel(flowComp,device_ids=device_ids)
flowComp.to(device)


ArbTimeFlowIntrp = UNet(20, 5)
ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
for param in ArbTimeFlowIntrp.parameters():
    param.requires_grad = False
# ArbTimeFlowIntrp = nn.DataParallel(ArbTimeFlowIntrp,device_ids=device_ids)
ArbTimeFlowIntrp.to(device)

flowBackWarp = backWarp(W, H, "cpu")
# flowBackWarp = nn.DataParallel(flowBackWarp,device_ids=device_ids)
flowBackWarp = flowBackWarp.to(device)

model_ckpts = [os.path.join(base_dir, expname, f) for f in sorted(os.listdir(os.path.join(base_dir, expname))) if 'tar' in f]
print('Found ckpts', model_ckpts)
if len(model_ckpts) > 0:
    model_ckpt_path = model_ckpts[-1]
    ckpt = torch.load(model_ckpt_path, map_location = device)
    global_step = ckpt['start_step'] + 1
else:
    ckpt = None
    global_step = 0

#load model if required	
focal_net = LearnFocal(H, W, req_grad=True).to(device)
pose_param_net = LearnPose(num_cams=N_IMGS, learn_R=True, learn_t=True).to(device)
time_pose_net = LearnTime(in_feat=3,mid_feat=32,out_feat=3).to(device)
nerf_model = Nerf(tpos_in_dims=126, dir_in_dims=27, D=256).to(device)

if ckpt is not None and load_model:

    focal_net.load_state_dict(ckpt['focal_net'])
    pose_param_net.load_state_dict(ckpt['pose_param_net'])
    time_pose_net.load_state_dict(ckpt['time_pose_net'])
    nerf_model.load_state_dict(ckpt['nerf_model'])


focal_net = nn.parallel.DistributedDataParallel(focal_net,
                                                device_ids=[args.local_rank],
                                                output_device=args.local_rank)
# focal_net.to(device)


# pose_param_net = nn.DataParallel(pose_param_net,device_ids=device_ids)
# pose_param_net.to(device) 
pose_param_net = nn.parallel.DistributedDataParallel(pose_param_net,
                                                device_ids=[args.local_rank],
                                                output_device=args.local_rank)

# time_pose_net = nn.DataParallel(time_pose_net,device_ids=device_ids)
# time_pose_net.to(device)
time_pose_net = nn.parallel.DistributedDataParallel(time_pose_net,
                                                device_ids=[args.local_rank],
                                                output_device=args.local_rank)
# Get a  NeRF model. Hidden dimension set to 256

# nerf_model = nn.DataParallel(nerf_model,device_ids=device_ids)
# nerf_model.to(device)
nerf_model = nn.parallel.DistributedDataParallel(nerf_model,
                                                device_ids=[args.local_rank],
                                                output_device=args.local_rank)

# Set lr and scheduler: these are just stair-case exponantial decay lr schedulers.
opt_nerf = torch.optim.Adam(nerf_model.parameters(), lr=0.001)
opt_focal = torch.optim.Adam(focal_net.parameters(), lr=0.001)
opt_pose = torch.optim.Adam(pose_param_net.parameters(), lr=0.001)
opt_time = torch.optim.Adam(time_pose_net.parameters(), lr=0.001)

scheduler_nerf = MultiStepLR(opt_nerf, milestones=list(range(0, 10000, 10)), gamma=0.9954)
scheduler_focal = MultiStepLR(opt_focal, milestones=list(range(0, 10000, 100)), gamma=0.9)
scheduler_pose = MultiStepLR(opt_pose, milestones=list(range(0, 10000, 100)), gamma=0.9)
scheduler_time = MultiStepLR(opt_time, milestones=list(range(0, 10000, 100)), gamma=0.9)

# Set tensorboard writer
# writer = SummaryWriter(log_dir=os.path.join('logs', expname, str(datetime.datetime.now().strftime('%y%m%d_%H%M%S'))))

# Training
print('Start Training...')
import time

for epoch_i in tqdm(range(global_step, N_EPOCH), desc='Training'):
    
    Tr_loss = train_one_epoch(image_data, H, W, ray_params, opt_nerf, opt_focal,opt_pose,opt_time, nerf_model, focal_net, pose_param_net,time_pose_net,flowComp,ArbTimeFlowIntrp,flowBackWarp)
    train_psnr = mse2psnr(Tr_loss)
 
    fxfy = de_parallel(focal_net)()
    #print('epoch {0:4d} Training PSNR {1:.3f}, estimated fx {2:.1f} fy {3:.1f}'.format(epoch_i, train_psnr, fxfy[0], fxfy[1]))
    print(f"epoch {epoch_i+1}: Training PSNR {train_psnr}, estimated fx {fxfy[0]} fy {fxfy[1]}")

    scheduler_nerf.step()
    scheduler_focal.step()
    scheduler_pose.step()
    scheduler_time.step()

    learned_c2ws = torch.stack([pose_param_net(i) for i in range(N_IMGS)])  # (N, 4, 4)
    
    #save check point

    if args.local_rank == 0:
        if (epoch_i) % SAVE_WEIGHTS_INTERVAL == 0:
            save_dict = {"start_step" : epoch_i,
                        "nerf_model": de_parallel(nerf_model).state_dict(),
                        "focal_net": de_parallel(focal_net).state_dict(),
                        "pose_param_net": de_parallel(pose_param_net).state_dict(),
                        "time_pose_net": de_parallel(time_pose_net).state_dict() }
            path = os.path.join(base_dir, expname, f'{epoch_i:06d}.tar')
            torch.save(save_dict, path)
            print("Save checkpoints at", path)
        # torch.save(nerf_model.module.state_dict(),f"{model_weight_dir}/{scene_name}_nerf.pt")
        # torch.save(focal_net.module.state_dict(),f"{model_weight_dir}/{scene_name}_focal.pt")
        # torch.save(pose_param_net.module.state_dict(),f"{model_weight_dir}/{scene_name}_pose.pt")
        # torch.save(time_pose_net.module.state_dict(),f"{model_weight_dir}/{scene_name}_time.pt")

        with torch.no_grad():  
            if (epoch_i) % EVAL_INTERVAL == 0:
                
                eval_c2w = torch.eye(4, dtype=torch.float32)  # (4, 4)
                fxfy = de_parallel(focal_net)()
                rendered_img, rendered_depth = render_novel_view(((epoch_i+1)%TSteps)/TSteps,eval_c2w, H, W, fxfy, ray_params, nerf_model,time_pose_net)
                imageio.imwrite(os.path.join(base_dir, expname, scene_name + f"_img{epoch_i+1}_SlMo.png"),(rendered_img*255).cpu().numpy().astype(np.uint8))
                imageio.imwrite(os.path.join(base_dir, expname, scene_name + f"_depth{epoch_i+1}_SlMoF.png"),(rendered_depth*200).cpu().numpy().astype(np.uint8))

            if (epoch_i) % SAVE_IMAGE_INTERVAL == 0:
                resize_ratio = 1
                num_steps = TSteps * 2             
                optimised_poses = torch.stack([pose_param_net(i) for i in range(N_IMGS)])
                radii = np.percentile(np.abs(optimised_poses.cpu().numpy()[:, :3, 3]), q=75, axis=0)  # (3,)
                spiral_c2ws = create_spiral_poses(radii, focus_depth=3.5, n_poses=num_steps, n_circle=1)
                spiral_c2ws = torch.from_numpy(spiral_c2ws).float()  # (N, 3, 4)

                # change intrinsics according to resize ratio
                fxfy = de_parallel(focal_net)()
                novel_fxfy = fxfy / resize_ratio
                novel_H, novel_W = H // resize_ratio, W // resize_ratio

                print('NeRF trained in {0:d} x {1:d} for {2:d} epochs'.format(H, W, N_EPOCH))
                print('Rendering novel views in {0:d} x {1:d}'.format(novel_H, novel_W))

                #time moment
                t = np.linspace(start=0,stop=1,num=num_steps,endpoint=True)
                
                novel_img_list, novel_depth_list = [], []

                #record processing time
                curr_time = time.time()

                #render images
                for i in tqdm(range(spiral_c2ws.shape[0]), desc='novel view rendering'):
                    
                    novel_img, novel_depth = render_novel_view(t[i],spiral_c2ws[i], novel_H, novel_W, novel_fxfy,ray_params, nerf_model,time_pose_net)
                    
                    novel_img_list.append(novel_img)
                    novel_depth_list.append(novel_depth)

                print('Novel view rendering done. Saving to GIF images...')
                print(f"It takes {time.time()-curr_time} to  complete the whole process")
                novel_img_list = (torch.stack(novel_img_list) * 255).cpu().numpy().astype(np.uint8)
                novel_depth_list = (torch.stack(novel_depth_list) * 200).cpu().numpy().astype(np.uint8)  # depth is always in 0 to 1 in NDC

                #os.makedirs('nvs_results', exist_ok=True)
                imageio.mimwrite(os.path.join(base_dir, expname, f'{epoch_i:06d}_img.gif'), novel_img_list, fps=30)
                imageio.mimwrite(os.path.join(base_dir, expname, f'{epoch_i:06d}_depth.gif'), novel_depth_list, fps=30)
                print('GIF images saved.')



print('Training Completed !!!')


# Novel View Synthesis
# Render novel views from a sprial camera trajectory.

if args.local_rank == 0:
    # Render full images are time consuming, especially on colab so we render a smaller version instead.
    resize_ratio = 1
    num_steps = TSteps * 2
    with torch.no_grad():
        
        optimised_poses = torch.stack([pose_param_net(i) for i in range(N_IMGS)])
        radii = np.percentile(np.abs(optimised_poses.cpu().numpy()[:, :3, 3]), q=75, axis=0)  # (3,)
        spiral_c2ws = create_spiral_poses(radii, focus_depth=3.5, n_poses=num_steps, n_circle=1)
        spiral_c2ws = torch.from_numpy(spiral_c2ws).float()  # (N, 3, 4)

        # change intrinsics according to resize ratio
        fxfy = de_parallel(focal_net)()
        novel_fxfy = fxfy / resize_ratio
        novel_H, novel_W = H // resize_ratio, W // resize_ratio

        print('NeRF trained in {0:d} x {1:d} for {2:d} epochs'.format(H, W, N_EPOCH))
        print('Rendering novel views in {0:d} x {1:d}'.format(novel_H, novel_W))

        #time moment
        t = np.linspace(start=0,stop=1,num=num_steps,endpoint=True)
        
        novel_img_list, novel_depth_list = [], []

        #record processing time
        curr_time = time.time()

        #render images
        for i in tqdm(range(spiral_c2ws.shape[0]), desc='novel view rendering'):
            
            novel_img, novel_depth = render_novel_view(t[i],spiral_c2ws[i], novel_H, novel_W, novel_fxfy,ray_params, nerf_model,time_pose_net)
            
            novel_img_list.append(novel_img)
            novel_depth_list.append(novel_depth)

        print('Novel view rendering done. Saving to GIF images...')
        print(f"It takes {time.time()-curr_time} to  complete the whole process")
        novel_img_list = (torch.stack(novel_img_list) * 255).cpu().numpy().astype(np.uint8)
        novel_depth_list = (torch.stack(novel_depth_list) * 200).cpu().numpy().astype(np.uint8)  # depth is always in 0 to 1 in NDC

        #os.makedirs('nvs_results', exist_ok=True)
        imageio.mimwrite(os.path.join(base_dir, expname, f'{epoch_i:06d}_img.gif'), novel_img_list, fps=30)
        imageio.mimwrite(os.path.join(base_dir, expname, f'{epoch_i:06d}_depth.gif'), novel_depth_list, fps=30)
        print('GIF images saved.')
