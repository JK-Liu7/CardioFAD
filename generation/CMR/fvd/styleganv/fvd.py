import torch
import os
import math
import torch.nn.functional as F

# https://github.com/universome/fvd-comparison


def load_i3d_pretrained(device=torch.device('cpu')):
    device = torch.device(device)
    i3D_WEIGHTS_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'i3d_torchscript.pt')
    print('Loading model from: %s'%filepath)
    if not os.path.exists(filepath):
        print(f"preparing for download {i3D_WEIGHTS_URL}, you can download it by yourself.")
        torch.hub.download_url_to_file(i3D_WEIGHTS_URL, filepath)
    i3d = torch.jit.load(filepath, map_location=device).eval().to(device)
    if device.type == 'cuda':
        index = device.index if device.index is not None else torch.cuda.current_device()
        i3d = torch.nn.DataParallel(i3d, device_ids=[index], output_device=index)
    return i3d


def get_feats(videos, detector, device, bs=10):

    detector_kwargs = dict(rescale=False, resize=False, return_features=True)
    feats = np.empty((0, 400))
    with torch.no_grad():
        for i in range((len(videos)-1)//bs + 1):
            feats = np.vstack([feats, detector(x=torch.stack([preprocess_single(video) for video in videos[i*bs:(i+1)*bs]]).to(device), **detector_kwargs).detach().cpu().numpy()])
    return feats


def get_fvd_feats(videos, i3d, device, bs=10):


    embeddings = get_feats(videos, i3d, device, bs)
    return embeddings


def preprocess_single(video, resolution=224, sequence_length=None):

    c, t, h, w = video.shape


    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:, :sequence_length]


    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear', align_corners=False)


    c, t, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]


    video = (video - 0.5) * 2

    return video.contiguous()


"""
Copy-pasted from https://github.com/cvpr2022-stylegan-v/stylegan-v/blob/main/src/metrics/frechet_video_distance.py
"""
from typing import Tuple
from scipy.linalg import sqrtm
import numpy as np


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def frechet_distance(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)
    m = np.square(mu_gen - mu_real).sum()
    if feats_fake.shape[0]>1:
        s, _ = sqrtm(np.dot(sigma_gen, sigma_real), disp=False)
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    else:
        fid = np.real(m)
    return float(fid)
