import torch
from tqdm import tqdm

def trans(x):

    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)


    x = x.permute(0, 2, 1, 3, 4)

    return x

def calculate_fvd(videos1, videos2, device, method='styleganv', only_final=False):

    if method == 'styleganv':
        from fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
    elif method == 'videogpt':
        from fvd.videogpt.fvd import load_i3d_pretrained, frechet_distance
        from fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats

    print("calculate_fvd...")


    assert videos1.shape == videos2.shape

    i3d = load_i3d_pretrained(device=device)
    fvd_results = []

    import torch.nn as nn
    if isinstance(i3d, nn.DataParallel):
        i3d = i3d.module

    i3d = i3d.to(device).eval()
    for p in i3d.parameters():
        p.requires_grad_(False)

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    fvd_results = []

    if only_final:

        assert videos1.shape[2] >= 10, "for calculate FVD, each clip_timestamp must >= 10"

        videos_clip1 = videos1
        videos_clip2 = videos2

        feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device, bs=20)
        feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device, bs=20)

        fvd_results.append(frechet_distance(feats1, feats2))

    else:
        for clip_timestamp in tqdm(range(10, videos1.shape[-3]+1)):

            videos_clip1 = videos1[:, :, : clip_timestamp]
            videos_clip2 = videos2[:, :, : clip_timestamp]

            feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
            feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)


            fvd_results.append(frechet_distance(feats1, feats2))

    result = {
        "value": fvd_results,
    }

    return result


def main():
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 20
    CHANNEL = 3
    SIZE = 128
    videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    device = torch.device("cuda")

    result = calculate_fvd(videos1, videos2, device, method='videogpt', only_final=True)
    print("[fvd-videogpt ]", result["value"])

    result = calculate_fvd(videos1, videos2, device, method='styleganv', only_final=True)
    print("[fvd-styleganv]", result["value"])

if __name__ == "__main__":
    main()
