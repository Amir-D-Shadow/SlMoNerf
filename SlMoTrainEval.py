import os, random, datetime
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
from helper import render_inp_frame

#setting
scene_name = "room6"
img_dir = f"{os.getcwd()}/data/{scene_name}"
model_weight_dir = f"{os.getcwd()}/model_weights/{scene_name}"

device = torch.device("cuda:6")
device_ids = [6,7]
num_of_device = len(device_ids)

SSIM_Metric = SSIM(size_average=True)
SSIM_Metric = nn.DataParallel(SSIM_Metric,device_ids=device_ids)
SSIM_Metric.to(device)

LIPIPS = VGGPerceptualLoss(resize=True,device=device)
LIPIPS = nn.DataParallel(LIPIPS,device_ids=device_ids)
LIPIPS.to(device)

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
        img = Image.fromarray(img).resize((512,384),Image.BILINEAR) #reshape (640,360)
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
        self.H = W#H
        self.W = W
        self.fx = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
        self.fy = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )

    def forward(self):
        # order = 2, check our supplementary.
        fxfy = torch.stack([self.fx**2 * self.W, self.fy**2 * self.H])
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
      self.act_fn1 = nn.LeakyReLU(negative_slope = 0.03)
      self.act_fn2 = nn.LeakyReLU(negative_slope = 0.03)


   def forward(self,x):

      """
      x (time tensor)  
      """

      timeStep = self.TimeLayer1(x)
      timeStep = self.act_fn1(timeStep)

      timeStep = self.TimeLayer2(timeStep)
      #(H,W,out_feat)
      final_output = self.act_fn2(timeStep)

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
      self.Time_ENC_FREQ = 4


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
    timeTensor = time_pose_net(timeTensor)                                    #(H,W,out_feat)
    timeTensor = timeTensor.unsqueeze(2).expand(-1,-1,ray_params.N_SAMPLE,-1) #(H,W,N_sample,out_feat)
    timeTensor = encode_position(timeTensor, levels=ray_params.Time_ENC_FREQ, inc_input=True) #(H,W,N_sample,27)
    
    # encode position: (H, W, N_sample, (2L+1)*C = 63)
    pos_enc = encode_position(sample_pos, levels=ray_params.POS_ENC_FREQ, inc_input=True)

    #cat spatial and time
    tpos_enc = torch.cat([pos_enc,timeTensor],dim=-1) #(H,W,N_sample,63+27)

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
def ModelEval(images_data, H, W, ray_params, nerf_model, focal_net, pose_param_net,time_pose_net):

    """
    images_data: {0:(N,H,W,C),1:(N,H,W,C),2:(N,H,W,C),...}   dictionary that stores images for each time step with N camera pose each
    """

    nerf_model.eval()
    focal_net.eval()
    pose_param_net.eval()
    time_pose_net.eval()

    #set up
    mean = [0.429, 0.431, 0.397]
    std  = [1, 1, 1]

    negmean = [x * -1 for x in mean]
    revNormalize = transforms.Normalize(mean=negmean, std=std)
    
    TP = transforms.Compose([revNormalize,transforms.ToPILImage()])

    t_vals = torch.linspace(ray_params.NEAR, ray_params.FAR, ray_params.N_SAMPLE, device=device)  # (N_sample,) sample position

    rgb_ssim_epoch = []
    rgb_lipips_epoch = []
    
    #shuffle the time steps
    time_list = [ i for i in range(TSteps)]

    for t in time_list:

        imgs0 = images_data[t].to(device)  #(N, 3, H, W)
        
        # arrange camera ids adn set time query
        ids = np.arange(N_IMGS)
        t_idx = (t / TSteps )

        #get images
        imgs = [] 
        for i in range(N_IMGS):
            imgs.append(TP(imgs0[i]))

        imgs = torch.from_numpy( np.stack(imgs,axis=0) ).float() #(N,H,W,3)

        #set up coordinate for patches
        row_size,col_size = 64,64 
        row_list = [(rs_i*row_size,(rs_i+1)*row_size) for rs_i in range(H//row_size)]
        col_list = [(cs_j*col_size,(cs_j+1)*col_size) for cs_j in range(W//col_size)]

        for row_idx in range(len(row_list)):

            for col_idx in range(len(col_list)):   

                for i in ids:

                    fxfy = focal_net()
                    
                    # KEY 1: compute ray directions using estimated intrinsics online.
                    ray_dir_cam = comp_ray_dir_cam_fxfy(H, W, fxfy[0], fxfy[1])
                    img = imgs[i].to(device)            # (H, W, 3)
                    c2w = pose_param_net(i)  # (4, 4)

                    # sample 32x32 pixel on an image and their rays for training.
                    #r_id = torch.randperm(H, device=device)[:90]  # (N_select_rows)
                    #c_id = torch.randperm(W, device=device)[:64]  # (N_select_cols)
                    row_start , row_end = row_list[row_idx]
                    col_start , col_end = col_list[col_idx]
                    ray_selected_cam = ray_dir_cam[row_start:row_end,col_start:col_end,:]  # (N_select_rows, N_select_cols, 3)
                    img_selected = img[row_start:row_end,col_start:col_end,:]  # (N_select_rows, N_select_cols, 3)

                    # render an image using selected rays, pose, sample intervals, and the network
                    render_result = model_render_image(time_pose_net,t_idx,c2w, ray_selected_cam, t_vals, ray_params,H, W, fxfy, nerf_model, perturb_t=True, sigma_noise_std=0.0)
                    rgb_rendered = render_result['rgb'] * 255.0  # (N_select_rows, N_select_cols, 3)

                    #SSIM
                    ssim_syn = rearrange( rgb_rendered.unsqueeze(0), "b h w c -> b c h w") # (1,N_select_rows, N_select_cols, 3) -> (1,3,N_select_rows, N_select_cols)
                    ssim_tgt = rearrange( img_selected.unsqueeze(0), "b h w c -> b c h w") # (1,N_select_rows, N_select_cols, 3) -> (1,3,N_select_rows, N_select_cols)
                    rgb_ssim =  SSIM_Metric(ssim_syn,ssim_tgt)

                    #LIPIPS
                    lipips_syn = rearrange( rgb_rendered.unsqueeze(0), "b h w c -> b c h w") # (1,N_select_rows, N_select_cols, 3) -> (1,3,N_select_rows, N_select_cols)
                    lipips_tgt = rearrange( img_selected.unsqueeze(0), "b h w c -> b c h w") # (1,N_select_rows, N_select_cols, 3) -> (1,3,N_select_rows, N_select_cols)
                    rgb_lipips = LIPIPS(lipips_syn/255.0,lipips_tgt/255.0)

                    with torch.no_grad():

                        rgb_ssim_epoch.append(rgb_ssim.clone().detach())
                        rgb_lipips_epoch.append(rgb_lipips.clone().detach())

                        print('Temporal SSIM {0:.2f}, Temporal LIPIPS {1:.2f}'.format(rgb_ssim.item(), rgb_lipips.item()))

    
    #summarize the metrics
    rgb_ssim_mean = torch.stack(rgb_ssim_epoch).mean().item()
    rgb_lipips_mean = torch.stack(rgb_lipips_epoch).mean().item()

    return rgb_ssim_mean,rgb_lipips_mean




#evalation
focal_net = LearnFocal(H, W, req_grad=True)
focal_net.load_state_dict(torch.load(f"{model_weight_dir}/{scene_name}_focal.pt",map_location = device))
for param in focal_net.parameters():
    param.requires_grad = False
focal_net = nn.DataParallel(focal_net,device_ids=device_ids)
focal_net.to(device)

pose_param_net = LearnPose(num_cams=N_IMGS, learn_R=True, learn_t=True)    
pose_param_net.load_state_dict(torch.load(f"{model_weight_dir}/{scene_name}_pose.pt",map_location = device))
for param in pose_param_net.parameters():
    param.requires_grad = False
pose_param_net = nn.DataParallel(pose_param_net,device_ids=device_ids)
pose_param_net.to(device) 

time_pose_net = LearnTime(in_feat=3,mid_feat=32,out_feat=3)
time_pose_net.load_state_dict(torch.load(f"{model_weight_dir}/{scene_name}_time.pt",map_location = device))
for param in time_pose_net.parameters():
    param.requires_grad = False
time_pose_net = nn.DataParallel(time_pose_net,device_ids=device_ids)
time_pose_net.to(device)


# Get a  NeRF model. Hidden dimension set to 256
nerf_model = Nerf(tpos_in_dims=90, dir_in_dims=27, D=256)
nerf_model.load_state_dict(torch.load(f"{model_weight_dir}/{scene_name}_nerf.pt",map_location = device))
for param in nerf_model.parameters():
    param.requires_grad = False
nerf_model = nn.DataParallel(nerf_model,device_ids=device_ids)
nerf_model.to(device)


# Evaluation
import json
print(f"Start Evaluation {scene_name} ...")

    
SSIM_info,LIPIPS_info = ModelEval(image_data, H, W, ray_params, nerf_model, focal_net, pose_param_net,time_pose_net)

print(f"Evaluating SSIM : {SSIM_info}, Evaluating LIPIPS : {LIPIPS_info}")


with torch.no_grad():
    
    file = open(os.path.join(f"{os.getcwd()}/nvs_midImg/{scene_name}", f"{scene_name}_record.txt"))
    c = json.load(file)

    file.close()

    PSNR_info = c["PSNR"]

    Tr_status = {"PSNR":str(PSNR_info),"SSIM":str(SSIM_info),"LIPIPS":str(LIPIPS_info)}
    with open(os.path.join(f"{os.getcwd()}/nvs_midImg/{scene_name}", f"{scene_name}_AdditionalRecord.txt"),"w") as file:

        file.write(json.dumps(Tr_status))

    file.close()


print('Evaluation Completed !!!')



