from pathlib import Path
import os
import torch
from lightglue import SuperPoint, LightGlue
from lightglue.utils import load_image, rbd
import pickle
import gzip
import cv2
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset

# Set environment variables and device
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load feature extractor and matcher
extractor = SuperPoint(max_num_keypoints=2000).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)

# Image directory path
images_dir = Path("/data/23_24")
image_files = list(images_dir.glob("*.png"))
L = len(image_files)
print(f"Found {len(image_files)} image files.")

# Save progress to a file
def save_progress(progress_file, start_i, completed_pairs):
    """Save progress to a file"""
    with open(progress_file, "wb") as pf:
        progress_data = {'start_i': start_i, 'completed_pairs': completed_pairs}
        pickle.dump(progress_data, pf)

# Output paths
output_file = Path("/result/r23.txt")
progress_file = Path("/process/r23.pkl")

# Cache dictionary for extracted features
features_cache = {}

def extract_and_cache_features(image_path, extractor, device):
    """Extract and cache features for an image"""
    if image_path in features_cache:
        return features_cache[image_path]
    image = load_image(image_path)
    image = image.to(device)
    feats = extractor.extract(image)
    features_cache[image_path] = feats
    return feats

# Load progress from a file
def load_progress(progress_file):
    """Load saved progress"""
    if progress_file.exists():
        with open(progress_file, "rb") as pf:
            progress_data = pickle.load(pf)
            start_i = progress_data.get('start_i', 0)
            completed_pairs = progress_data.get('completed_pairs', set())
        return start_i, completed_pairs
    else:
        return 0, set()

# Save a matching result to a file
def save_matching_result(base_img, compare_img_path, matches, output_file):
    """Save matching result to a file"""
    if len(matches) > 0:
        with open(output_file, "a") as f:
            f.write(f"{base_img.name}, {compare_img_path.name}, Matches: {len(matches)}\n")
        print(f"Saved match: {base_img.name} and {compare_img_path.name}")

# Resume from saved progress
if progress_file.exists():
    start_i, completed_pairs = load_progress(progress_file)
    print(f"Resumed progress: start_i={start_i}, completed pairs={len(completed_pairs)}")
else:
    start_i = 0
    completed_pairs = set()

# Compute total number of image pairs
total_pairs = sum(1 for i in range(len(image_files)) for j in range(i + 1, len(image_files)))
print(f"Total number of image pairs: {total_pairs}")

# Preload all features
def load_all_features(image_files, extractor, device):
    all_features = {}
    for image_path in tqdm(image_files, desc="Loading and Extracting Features"):
        feats = extract_and_cache_features(image_path, extractor, device)
        all_features[image_path] = feats
    return all_features

all_features = load_all_features(image_files, extractor, device)

# Dataset for base image
class BaseImageDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        return load_image(image_path)

# Dataset for image pairs
class PairDataset(Dataset):
    def __init__(self, base_image, compare_images):
        self.base_image = base_image
        self.compare_images = compare_images

    def __len__(self):
        return len(self.compare_images)

    def __getitem__(self, idx):
        compare_image_path = self.compare_images[idx]
        compare_image = load_image(compare_image_path)
        return self.base_image, compare_image, compare_image_path

# Compute homography and warp image
def compute_homography_and_warp(image1, image2, kpts1, kpts2, matches):
    if matches.size == 0:
        return None, None, None
    points1 = np.float32([kpts1[m[0]].cpu().numpy() for m in matches])
    points2 = np.float32([kpts2[m[1]].cpu().numpy() for m in matches])
    if len(points1) < 4 or len(points2) < 4:
        return None, None, None
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    if H is None:
        return None, None, None
    height, width = image2.shape[:2]
    warped_image = cv2.warpPerspective(image1, H, (width, height))
    if warped_image.ndim == 2:
        warped_image = warped_image[:, :, np.newaxis]
    return warped_image, H, mask

# DataLoader for base images
base_dataset = BaseImageDataset(image_files)
base_dataloader = DataLoader(base_dataset, batch_size=1, shuffle=False)

# Main loop
try:
    with open(output_file, "a") as f, tqdm(total=total_pairs, desc="Processing Image Pairs") as pbar:
        for i, base_image_tensor in enumerate(base_dataloader):
            if i < start_i:
                pbar.update(len(image_files) - i - 1)
                continue
            base_image_path = image_files[i]
            if i < L:
                compare_images_path = image_files[i+1]
            base_image_tensor = base_image_tensor[0].to(device, non_blocking=True)
            print(f"Processing base image {image_files[i].name}")
            compare_images = image_files[i + 1:]
            pair_dataset = PairDataset(base_image_tensor, compare_images)
            pair_dataloader = DataLoader(pair_dataset, batch_size=3000, num_workers=0, collate_fn=lambda x: x)

            for batch_index, batch in enumerate(pair_dataloader):
                for base_image, compare_image, compare_image_path in batch:
                    compare_image = compare_image.to(device, non_blocking=True)
                    pair_id = (i, image_files.index(compare_image_path))
                    if pair_id in completed_pairs:
                        pbar.update(1)
                        continue

                    with autocast():
                        feats0 = all_features[base_image_path]
                        feats1 = all_features[compare_image_path]
                        matches01 = matcher({"image0": feats0, "image1": feats1})

                    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
                    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]

                    match_ratio_0 = len(matches) / len(kpts0) if len(kpts0) > 0 else 0
                    match_ratio_1 = len(matches) / len(kpts1) if len(kpts1) > 0 else 0
                    max_match_ratio = max(match_ratio_0, match_ratio_1)

                    if max_match_ratio > 0.20:
                        print(f"Match found between {image_files[i].name} and {compare_image_path.name} with max ratio {max_match_ratio:.2f}")
                        image0_np = base_image.permute(1, 2, 0).cpu().numpy()
                        image1_np = compare_image.permute(1, 2, 0).cpu().numpy()

                        if len(kpts0) <= len(kpts1):
                            src_image = image0_np
                            dst_image = image1_np
                            src_kpts = kpts0
                            dst_kpts = kpts1
                            src_image_tensor = base_image
                            dst_image_tensor = compare_image
                        else:
                            src_image = image1_np
                            dst_image = image0_np
                            src_kpts = kpts1
                            dst_kpts = kpts0
                            src_image_tensor = compare_image
                            dst_image_tensor = base_image

                        adjusted_matches = matches.clone()
                        if len(kpts0) > len(kpts1):
                            adjusted_matches = adjusted_matches[:, [1, 0]]

                        warped_image, H, mask = compute_homography_and_warp(src_image, dst_image, src_kpts, dst_kpts, adjusted_matches)

                        if warped_image is not None:
                            if warped_image.ndim == 2:
                                warped_image = warped_image[:, :, np.newaxis]
                            warped_image_tensor = torch.tensor(warped_image).permute(2, 0, 1).unsqueeze(0).float().to(device)
                            feats_warped = extractor.extract(warped_image_tensor)
                            feats_dst_new = extractor.extract(dst_image_tensor.unsqueeze(0))
                            matches_warped = matcher({"image0": feats_warped, "image1": feats_dst_new})
                            feats_warped, feats_dst_new, matches_warped = [rbd(x) for x in [feats_warped, feats_dst_new, matches_warped]]
                            match_rate_2 = len(matches_warped["matches"]) / len(feats_warped["keypoints"]) if len(feats_warped["keypoints"]) > 0 else 0

                            if match_rate_2 > max_match_ratio:
                                f.write(f"{image_files[i].name}, {compare_image_path.name}: {match_rate_2:.2f}\n")
                                f.flush()
                                print(f"Match saved between {image_files[i].name} and {compare_image_path.name} with best ratio {match_rate_2:.2f}")

                    completed_pairs.add(pair_id)
                    pbar.update(1)
                    if pbar.n % 100 == 0:
                        save_progress(progress_file, i, completed_pairs)

                torch.cuda.empty_cache()

except Exception as e:
    print(f"An error occurred: {e}")

save_progress(progress_file, len(image_files), completed_pairs)
print(f"Matched pairs saved to {output_file}")
