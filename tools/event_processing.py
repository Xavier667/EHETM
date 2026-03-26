import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

def load_frame(filepath, crop_size=512):
    """
    Load a single event frame and center-crop.
    """
    img = Image.open(filepath).convert('L')
    W, H = img.size
    start_H = (H - crop_size) // 2
    start_W = (W - crop_size) // 2
    img_crop = img.crop((start_W, start_H, start_W + crop_size, start_H + crop_size))
    return np.array(img_crop, dtype=np.uint8)

def to_binary_event(frame):
    """
    Convert frame to binary event (0 for no event, 1 for any event), ignoring polarity.
    """
    return (frame != 0).astype(np.float32)

def build_voxel_nonpolarity(event_imgs, accum_size=5):
    """
    Build voxel by accumulating every `accum_size` frames (ignoring polarity).
    Output shape: [T_voxel, H, W]
    """
    voxel_imgs = []
    for i in range(0, len(event_imgs), accum_size):
        batch = event_imgs[i:i + accum_size]
        voxel_imgs.append(np.sum(batch, axis=0))
    return np.stack(voxel_imgs, axis=0).astype(np.float32)

def build_voxel_polarity(event_imgs):
    """
    Build polarity-aware voxel.
    Mapping:
        200 -> +1 (positive event)
        100 -> -1 (negative event)
        0   -> 0 (no event)
    Output shape: [T,H,W]
    """
    voxel_ours = np.zeros((len(event_imgs), *event_imgs[0].shape), dtype=np.float32)
    for i, frame in enumerate(event_imgs):
        voxel_ours[i] = np.where(frame == 200, 1.0,
                          np.where(frame == 100, -1.0, 0.0))
    return voxel_ours

def compute_change_count(frames):
    """
    Count the number of polarity changes per pixel.
    Includes all changes (0->±1, ±1->0, ±1->∓1).
    Returns:
        change_count: (H,W), uint16
    """
    # Compute adjacent frame difference
    flips = np.diff(frames, axis=0)
    # Any non-zero difference counts as a change
    change_mask = flips != 0
    # Sum changes along time axis
    change_count = np.sum(change_mask, axis=0)
    return change_count.astype(np.uint16)

def process_event_folder(event_folder, save_folder, crop_size=512, frames_per_voxel=35):
    """
    Process event frames in chunks (e.g., 35 frames per sample).
    Each chunk generates one voxel and one PAEP map.
    """
    os.makedirs(save_folder, exist_ok=True)

    event_files = sorted(os.listdir(event_folder))
    total_frames = len(event_files)

    assert total_frames >= frames_per_voxel, "Not enough frames"

    num_samples = total_frames // frames_per_voxel

    for idx in range(num_samples):
        start = idx * frames_per_voxel
        end = start + frames_per_voxel
        batch_files = event_files[start:end]

        # --------- load frames ---------
        event_imgs = [
            load_frame(os.path.join(event_folder, f), crop_size)
            for f in batch_files
        ]

        # --------- voxel (no polarity) ---------
        binary_events = [to_binary_event(img) for img in event_imgs]
        voxel_nonpolarity = build_voxel_nonpolarity(binary_events)

        # --------- voxel (with polarity) ---------
        voxel_polarity = build_voxel_polarity(event_imgs)

        # --------- PAEP ---------
        polarity_alternation_count = compute_change_count(voxel_polarity)

        # --------- save (use index) ---------
        np.save(os.path.join(save_folder, f"{idx:06d}_voxel_nonpolarity.npy"), voxel_nonpolarity)
        np.save(os.path.join(save_folder, f"{idx:06d}_voxel_polarity.npy"), voxel_polarity)
        np.save(os.path.join(save_folder, f"{idx:06d}_PAEP.npy"), polarity_alternation_count)

        # visualization
        if np.max(polarity_alternation_count) > 0:
            vis = (255 * polarity_alternation_count / np.max(polarity_alternation_count)).astype(np.uint8)
        else:
            vis = np.zeros_like(polarity_alternation_count, dtype=np.uint8)

        cv2.imwrite(os.path.join(save_folder, f"{idx:06d}_PAEP.png"), vis)

    print(f"✅ {event_folder}: Generated {num_samples} samples")

def main():
    """
    Batch process multiple event folders.
    """
    root_event_dir = r"*\CTTH\Static\Test"  # Root folder containing event folders
    save_root = r"*\CTTH\event_processed"           # Root folder to save outputs
    crop_size = 512
    slice_per_voxel = 35

    subfolders = sorted(os.listdir(root_event_dir))
    for sub in tqdm(subfolders, desc="Processing cases"):
        event_folder = os.path.join(root_event_dir, sub, "event")  # Event folder per case
        save_folder = os.path.join(save_root, sub)
        if not os.path.exists(event_folder):
            print(f"⚠️ Skipping {sub}: event folder not found")
            continue
        process_event_folder(event_folder, save_folder, crop_size, slice_per_voxel)

if __name__ == "__main__":
    main()
