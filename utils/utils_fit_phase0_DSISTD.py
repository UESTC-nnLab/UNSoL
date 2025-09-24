import os
import torch
from tqdm import tqdm
import cv2
import numpy as np
from collections import defaultdict,  OrderedDict, deque
import math
import random
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from PIL import Image
import torchvision.transforms as transforms
from scipy.stats import entropy
import shutil
from typing import Dict, List, Tuple, Callable



path_file = '/home/public/DSISTD/train.txt'
final_file = '/home/public/DSISTD/train_pseudo_labels.txt'
candidate_file = '/home/public/DSISTD/candidate_pseudo_labels.txt'
aug_file = '/home/public/DSISTD/train_pseudo_labels_0.txt'
aug_dir = '/home/public/DSISTD_AUG'

def generation():
    with open(candidate_file, 'w') as f:
        pass
    with open(final_file, 'w') as f:
        pass

    frame_data = generate_pseudo_labels(path_file)
    print("-------Pseudo-label generation-------")
    frame_data = post_process_pseudo_labels(frame_data)
    print("-------Post-processing completed-------")
    label_m = filter_overlapping_boxes(frame_data, path_file)
    print("-------Pseudo-labels filtering completed-------")
    label_s = generate_label_s(path_file)
    dataset_complexity = compute_dataset_background_complexity(path_file)
    print(dataset_complexity)
    merge_pseudo_labels(dataset_complexity, label_m, label_s, path_file, final_file)
    filter_targets(final_file)     
    print("Augmenting pseudo-labels...")
    homogeneous_motion_hybridizing(final_file,aug_dir,aug_file,2)


def filter_targets(input_file: str, dist_thresh=30, min_track_len=7):
    def box_center(xmin, ymin, xmax, ymax):
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        return cx, cy

    def is_similar(box1, box2, dist_thresh=20):
        (x1min, y1min, x1max, y1max, c1) = box1
        (x2min, y2min, x2max, y2max, c2) = box2
        cx1, cy1 = box_center(x1min, y1min, x1max, y1max)
        cx2, cy2 = box_center(x2min, y2min, x2max, y2max)
        dist = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        return dist <= dist_thresh

    frames = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ')
            img_path = parts[0]
            boxes = parts[1:] if len(parts) > 1 else []
            box_list = []
            for b in boxes:
                coords = b.split(',')
                if len(coords) == 5:
                    xmin, ymin, xmax, ymax, cls = coords
                    xmin = int(xmin)
                    ymin = int(ymin)
                    xmax = int(xmax)
                    ymax = int(ymax)
                    cls = int(cls)
                    box_list.append((xmin, ymin, xmax, ymax, cls))
            frames.append((img_path, box_list))

    num_frames = len(frames)

    tracks = []
    for i, (img_path, box_list) in enumerate(frames):
        if i == 0:
            for box in box_list:
                tracks.append([(i, box)])
        else:
            used = [False]*len(box_list)
            prev_frame_idx = i - 1
            last_boxes = [(tid, tr[-1]) for tid, tr in enumerate(tracks) if tr[-1][0] == prev_frame_idx]

            for t_id, (prev_idx, prev_box) in last_boxes:
                best_match_id = -1
                best_dist = float('inf')
                px1, py1 = box_center(*prev_box[:4])

                for b_id, box in enumerate(box_list):
                    if used[b_id]:
                        continue
                    if is_similar(prev_box, box, dist_thresh=dist_thresh):
                        x2, y2 = box_center(*box[:4])
                        dist = math.sqrt((px1 - x2)**2 + (py1 - y2)**2)
                        if dist < best_dist:
                            best_dist = dist
                            best_match_id = b_id

                if best_match_id != -1:
                    tracks[t_id].append((i, box_list[best_match_id]))
                    used[best_match_id] = True
            for b_id, box in enumerate(box_list):
                if not used[b_id]:
                    tracks.append([(i, box)])
    valid_tracks = [tr for tr in tracks if len(tr) >= min_track_len]

    frame_dict = {i: [] for i in range(num_frames)}
    for tr in valid_tracks:
        for f_idx, box in tr:
            frame_dict[f_idx].append(box)

    filtered_frames = []
    for i, (img_path, box_list) in enumerate(frames):
        final_boxes = frame_dict[i]
        filtered_frames.append((img_path, final_boxes))

    with open(input_file, 'w') as f:
        for img_path, box_list in filtered_frames:
            if len(box_list) == 0:
                f.write(f"{img_path}\n")
            else:
                line = img_path
                for (xmin, ymin, xmax, ymax, cls) in box_list:
                    line += f" {xmin},{ymin},{xmax},{ymax},{cls}"
                f.write(line + "\n")

    print("Filtering complete. Results saved to:", input_file)



def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CustomDataset(Dataset):
    def __init__(self, txt_file):
        self.data = []
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if ' ' in line:
                parts = line.split(' ')
                img_path = parts[0]
                bbox = list(map(int, parts[1].split(',')))
                self.data.append((img_path, bbox))
            else:
                img_path = line
                self.data.append((img_path, None))
        self.img_transform = transforms.ToTensor()

    def __getitem__(self, index):
        img_path, bbox = self.data[index]
        img = Image.open(img_path).convert('L')
        original_size = img.size
        img = self.img_transform(img)
        return img, img_path, original_size

    def __len__(self):
        return len(self.data)

    @property
    def name(self):
        return 'DSISTD'

class S_Branch:
    def __init__(self, K=3, device='cuda'):
        self.K = K
        self.N = 5 * K
        self.device = device
        self.sigma = 1e-5
        self.prepare_kernels()

    def prepare_kernels(self):
        K, N = self.K, self.N
        op_in = torch.ones((1, 1, K, K), device=self.device)
        op_in_m = op_in / op_in.sum()
        self.op_in_m = op_in_m

        op_mid = torch.zeros((1, 1, N, N), device=self.device)
        op_mid[:, :, K:4*K, K:4*K] = 1
        op_mid[:, :, 2*self.K:3*self.K, 2*self.K:3*self.K] = 0
        op_mid_m = op_mid / op_mid.sum()
        self.op_mid_m = op_mid_m

        op_out = torch.ones((1, 1, N, N), device=self.device)
        op_out[:, :, K:4*K, K:4*K] = 0
        op_out_m = op_out / op_out.sum()
        self.op_out_m = op_out_m

        self.pad_in = (self.K // 2, self.K // 2, self.K // 2, self.K // 2)
        self.pad_mid = (self.N // 2, self.N // 2, self.N // 2, self.N // 2)
        self.pad_out = (self.N // 2, self.N // 2, self.N // 2, self.N // 2)

    def run_S_Branch(self, img):
        if len(img.shape) != 4:
            raise ValueError(f"Expected 4D input, got {img.shape}")

        img_padded_in = F.pad(img, self.pad_in, mode='replicate')
        img_mean_in = F.conv2d(img_padded_in, self.op_in_m)

        img_padded_mid = F.pad(img, self.pad_mid, mode='replicate')
        img_mean_mid = F.conv2d(img_padded_mid, self.op_mid_m)

        img_padded_out = F.pad(img, self.pad_out, mode='replicate')
        img_mean_out = F.conv2d(img_padded_out, self.op_out_m)

        out1 = (img_mean_in - img_mean_mid) + (img_mean_in - img_mean_out)
        out1 = torch.clamp(out1, min=0)

        op_mid_variance = torch.zeros((1, 1, self.N, self.N), device=self.device)
        op_mid_variance[:, :, self.K:4*self.K, self.K:4*self.K] = 1
        op_mid_variance[:, :, 2*self.K:3*self.K, 2*self.K:3*self.K] = 1
        op_mid_variance = op_mid_variance / op_mid_variance.sum()

        img_mean_mid_var = F.conv2d(F.pad(img, self.pad_mid, mode='replicate'), op_mid_variance)
        img_mean_sq_mid = F.conv2d(F.pad(img ** 2, self.pad_mid, mode='replicate'), op_mid_variance)
        img_std_mid = torch.sqrt(torch.clamp(img_mean_sq_mid - img_mean_mid_var ** 2, min=0))

        img_std_out = F.conv2d(F.pad(img ** 2, self.pad_out, mode='replicate'), self.op_out_m)
        img_std_out = torch.sqrt(torch.clamp(img_std_out - (F.conv2d(F.pad(img, self.pad_out, mode='replicate'), self.op_out_m)) ** 2, min=0))

        out2 = (img_std_mid / (img_std_out + self.sigma)) * img_std_mid - img_std_mid
        out2 = torch.clamp(out2, min=0)
        out = out1 * out2

        return out

class SimpleTracker:
    def __init__(self, max_age=30, distance_threshold=10):
        self.max_age = max_age
        self.distance_threshold = distance_threshold
        self.next_id = 1
        self.tracks = {}

    def update(self, detections):
        updated_tracks = []
        assigned_detections = set()
        current_centers = []

        for det in detections:
            x_min, y_min, x_max, y_max = det[:4]
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            current_centers.append((center_x, center_y))

        if len(self.tracks) == 0:
            for det in detections:
                self.tracks[self.next_id] = {
                    'bbox': det[:4].tolist(),
                    'center': self.compute_center(det[:4]),
                    'age': 0
                }
                updated_tracks.append([*det[:4], self.next_id])
                self.next_id += 1
        else:
            track_ids = list(self.tracks.keys())
            track_centers = [self.tracks[tid]['center'] for tid in track_ids]

            distance_matrix = self.compute_distance_matrix(track_centers, current_centers)
            track_matched = set()
            detection_matched = set()

            sorted_indices = np.argsort(distance_matrix, axis=None)
            for idx in sorted_indices:
                track_idx = idx // len(detections)
                det_idx = idx % len(detections)

                if track_idx in track_matched or det_idx in detection_matched:
                    continue

                distance = distance_matrix[track_idx, det_idx]
                if distance < self.distance_threshold:
                    track_id = track_ids[track_idx]
                    det = detections[det_idx]
                    self.tracks[track_id]['bbox'] = det[:4].tolist()
                    self.tracks[track_id]['center'] = self.compute_center(det[:4])
                    self.tracks[track_id]['age'] = 0
                    updated_tracks.append([*det[:4], track_id])
                    track_matched.add(track_idx)
                    detection_matched.add(det_idx)

            for det_idx, det in enumerate(detections):
                if det_idx not in detection_matched:
                    self.tracks[self.next_id] = {
                        'bbox': det[:4].tolist(),
                        'center': self.compute_center(det[:4]),
                        'age': 0
                    }
                    updated_tracks.append([*det[:4], self.next_id])
                    self.next_id += 1

            to_delete = []
            for track_idx, tid in enumerate(track_ids):
                if track_idx not in track_matched:
                    self.tracks[tid]['age'] += 1
                    if self.tracks[tid]['age'] > self.max_age:
                        to_delete.append(tid)
            for tid in to_delete:
                del self.tracks[tid]

        return np.array(updated_tracks)

    @staticmethod
    def compute_center(bbox):
        x_min, y_min, x_max, y_max = bbox
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        return (center_x, center_y)

    @staticmethod
    def compute_distance_matrix(track_centers, detection_centers):
        distance_matrix = np.zeros((len(track_centers), len(detection_centers)), dtype=np.float32)
        for t, (tx, ty) in enumerate(track_centers):
            for d, (dx, dy) in enumerate(detection_centers):
                distance = math.sqrt((tx - dx) ** 2 + (ty - dy) ** 2)
                distance_matrix[t, d] = distance
        return distance_matrix

class Inference:
    def __init__(self, device, K=3):
        self.device = device
        self.K = K
        self.S_Generation = S_Branch(K=self.K, device=self.device)
        self.tracker = SimpleTracker(max_age=30, distance_threshold=10)
        self.ids = []
        self.trajs = []
        self.frame_to_img_path = []
        self.frame_count = 0

    def __call__(self, img, img_path, original_size):
        if len(img.shape) == 5 and img.shape[1] == 1:
            img = img.squeeze(1)

        img = img.to(self.device)
        with torch.no_grad():
            out = self.S_Generation.run_S_Branch(img)

        out_np = out.squeeze().cpu().numpy()
        out_np = np.nan_to_num(out_np)
        out_norm = (out_np - out_np.min()) / (out_np.max() - out_np.min() + 1e-8)
        threshold = 0.5
        out_binary = np.zeros_like(out_norm)
        out_binary[out_norm >= threshold] = 1

        bboxes = self.mask_to_bboxes(out_binary)
        bboxes = self.merge_bboxes(bboxes, distance_threshold=4)

        if bboxes is not None:
            dets = np.array([bbox + [1.0] for bbox in bboxes])
        else:
            dets = np.empty((0, 5))

        track_bbs_ids = self.tracker.update(dets)
        for it in range(track_bbs_ids.shape[0]):
            track_id = int(track_bbs_ids[it, -1])
            coord = track_bbs_ids[it, :4]
            if track_id not in self.ids:
                self.ids.append(track_id)
                self.trajs.append([])
            index = self.ids.index(track_id)
            self.trajs[index].append([self.frame_count] + coord.tolist())

        self.frame_to_img_path.append(img_path)
        self.frame_count += 1

    def mask_to_bboxes(self, mask, max_bboxes=1):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append([x, y, x + w + 4, y + h + 4])
        if bboxes:
            bboxes = sorted(
                bboxes,
                key=lambda box: (box[2]-box[0])*(box[3]-box[1]),
                reverse=True
            )
            if max_bboxes is not None:
                bboxes = bboxes[:max_bboxes]
        return bboxes if bboxes else None

    def merge_bboxes(self, bboxes, distance_threshold=6):
        if bboxes is None:
            return None
        merged_bboxes = []
        for bbox in bboxes:
            x_min1, y_min1, x_max1, y_max1 = bbox
            merged = False
            for j, merged_bbox in enumerate(merged_bboxes):
                x_min2, y_min2, x_max2, y_max2 = merged_bbox
                center1 = ((x_min1 + x_max1)/2, (y_min1+y_max1)/2)
                center2 = ((x_min2 + x_max2)/2, (y_min2+y_max2)/2)
                distance = math.sqrt((center1[0]-center2[0])**2 + (center1[1]-center2[1])**2)
                if distance < distance_threshold:
                    new_x_min = min(x_min1,x_min2)
                    new_y_min = min(y_min1,y_min2)
                    new_x_max = max(x_max1,x_max2)
                    new_y_max = max(y_max1,y_max2)
                    merged_bboxes[j] = [new_x_min,new_y_min,new_x_max,new_y_max]
                    merged = True
                    break
            if not merged:
                merged_bboxes.append(bbox)
        return merged_bboxes

    def finalize(self):
        trajs_filt = []
        valid_track_ids = []
        for traj_i, track_id in zip(self.trajs, self.ids):
            if len(traj_i) < 7:
                continue
            a = np.array(traj_i)
            ct = (a[:, 1:3] + a[:, 3:5]) / 2
            d = ct[1:, :] - ct[:-1, :]
            d = np.sqrt(d[:,0]**2 + d[:,1]**2)
            v = d/(1+1e-6)
            v_mean = np.abs(v).mean()
            if v_mean < 0.55:
                continue
            trajs_filt.append(traj_i)
            valid_track_ids.append(track_id)

        self.ids = valid_track_ids
        self.trajs = trajs_filt

        det_for_images = [[] for _ in range(self.frame_count)]
        count = 0
        for i_traj in trajs_filt:
            count += 1
            for i_trajkk in i_traj:
                frame_idx = int(i_trajkk[0])
                det_for_images[frame_idx].append(i_trajkk[1:] + [count])

        label_s = {}
        for frame_idx, detections in enumerate(det_for_images):
            img_path = self.frame_to_img_path[frame_idx]
            if len(detections) > 0:
                boxes = []
                for x_min,y_min,x_max,y_max,track_id in detections:
                    boxes.append((int(x_min), int(y_min), int(x_max), int(y_max), 0))
                label_s[img_path] = boxes
            else:
                label_s[img_path] = []

        return label_s

def generate_label_s(path_file):
    set_seed(1024)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = CustomDataset(txt_file=path_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    inference = Inference(device=device, K=3)
    for data, img_path, original_size in tqdm(dataloader, desc="Processing dataset"):
        img_path_str = img_path[0]
        inference(data, img_path_str, original_size)
    label_s = inference.finalize()
    return label_s
    


def compute_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = map(int, box1[:4])
    x2_min, y2_min, x2_max, y2_max = map(int, box2[:4])

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    iou = inter_area / float(area1 + area2 - inter_area)
    return iou

def average_bboxes(background_complexity_coefficient, bbox1, bbox2):
    x_min1, y_min1, x_max1, y_max1 = bbox1[:4]
    x_min2, y_min2, x_max2, y_max2 = bbox2[:4]

    x_center1 = (x_min1 + x_max1) / 2.0
    y_center1 = (y_min1 + y_max1) / 2.0
    x_center2 = (x_min2 + x_max2) / 2.0
    y_center2 = (y_min2 + y_max2) / 2.0

    x_center_avg = (x_center1 + x_center2) / 2.0
    y_center_avg = (y_center1 + y_center2) / 2.0

    width1 = x_max1 - x_min1
    height1 = y_max1 - y_min1
    width2 = x_max2 - x_min2
    height2 = y_max2 - y_min2

    if background_complexity_coefficient > 0.5:
        width_avg = width2
        height_avg = height2
    else:
        width_avg = width1
        height_avg = height1

    x_min_avg = int(round(x_center_avg - width_avg / 2.0))
    y_min_avg = int(round(y_center_avg - height_avg / 2.0))
    x_max_avg = int(round(x_center_avg + width_avg / 2.0))
    y_max_avg = int(round(y_center_avg + height_avg / 2.0))

    return (x_min_avg, y_min_avg, x_max_avg, y_max_avg)

def match_bboxes_in_order(bboxes_m, bboxes_s):
    matched_pairs = []
    matched_s_indices = set()

    for bbox_m in bboxes_m:
        best_iou = 0.0
        best_s_idx = -1
        for idx, bbox_s in enumerate(bboxes_s):
            if idx in matched_s_indices:
                continue
            iou = compute_iou(bbox_m, bbox_s)
            if iou > best_iou:
                best_iou = iou
                best_s_idx = idx
        if best_iou > 0.0 and best_s_idx != -1:
            matched_pairs.append((bbox_m, bboxes_s[best_s_idx], best_iou))
            matched_s_indices.add(best_s_idx)
        else:
            matched_pairs.append((bbox_m, None, 0.0))

    unmatched_s = [bbox_s for idx, bbox_s in enumerate(bboxes_s) if idx not in matched_s_indices]
    return matched_pairs, unmatched_s

def merge_pseudo_labels(
    background_complexity_coefficient,
    label_m,
    label_s,
    path_file,
    final_file_path
):
    all_image_paths = []
    with open(path_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                image_path = line.split(' ')[0]
                all_image_paths.append(image_path)
    frames_m = set(label_m.keys())
    frames_s = set(label_s.keys())
    frames_s_only = [p for p in frames_s if p not in frames_m]
    
    with open(final_file_path, 'w') as f_out:
        for image_path in all_image_paths:
            bboxes_m = label_m.get(image_path, [])
            bboxes_s = label_s.get(image_path, [])
            final_bboxes = []

            if bboxes_m and bboxes_s:
                matched_pairs, unmatched_s = match_bboxes_in_order(bboxes_m, bboxes_s)

                for bbox_m, bbox_s, iou in matched_pairs:
                    if bbox_s is not None:
                        if background_complexity_coefficient > 0.5:
                            if iou > 0.7:
                                bbox_avg = average_bboxes(background_complexity_coefficient, bbox_m, bbox_s)
                                final_bboxes.append(bbox_avg)
                            else:
                                bbox_s_int = tuple(int(round(coord)) for coord in bbox_s[:4])
                                final_bboxes.append(bbox_s_int)
                        else:
                            if iou > 0.7:
                                bbox_avg = average_bboxes(background_complexity_coefficient, bbox_m, bbox_s)
                                final_bboxes.append(bbox_avg)
                            else:
                                bbox_m_int = tuple(int(round(coord)) for coord in bbox_m)
                                final_bboxes.append(bbox_m_int)
                    else:
                        if background_complexity_coefficient <= 0.5:
                            bbox_m_int = tuple(int(round(coord)) for coord in bbox_m)
                            final_bboxes.append(bbox_m_int)
                        else:
                            pass

                if background_complexity_coefficient > 0.5:
                    for bbox_s in unmatched_s:
                        bbox_s_int = tuple(int(round(coord)) for coord in bbox_s[:4])
                        final_bboxes.append(bbox_s_int)

            elif bboxes_s:
                if background_complexity_coefficient > 0.5:
                    for bbox_s_ in bboxes_s:
                        bbox_s_int = tuple(int(round(coord)) for coord in bbox_s_[:4])
                        final_bboxes.append(bbox_s_int)

            elif bboxes_m:
                if background_complexity_coefficient <= 0.5:
                    for bbox_m_ in bboxes_m:
                        bbox_m_int = tuple(int(round(coord)) for coord in bbox_m_)
                        final_bboxes.append(bbox_m_int)

            if final_bboxes:
                bboxes_str = ' '.join([f"{b[0]},{b[1]},{b[2]},{b[3]},0" for b in final_bboxes])
                line = f"{image_path} {bboxes_str}\n"
            else:
                line = f"{image_path}\n"

            f_out.write(line)


def compute_shannon_entropy(image):
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 255))
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    ent = entropy(prob, base=2)
    return ent

def compute_dataset_background_complexity(path_file):
    image_paths = []
    with open(path_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                image_path = parts[0]
                image_paths.append(image_path)
    
    entropies = []
    for img_path in tqdm(image_paths, desc="Processing Images"):
        if not os.path.exists(img_path):
            continue
        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img_resized = cv2.resize(img, (512, 512))
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            ent = compute_shannon_entropy(gray)
            entropies.append(ent)
        except Exception as e:
            continue
    
    if not entropies:
        print("No valid images found. Cannot compute background complexity.")
        return None
    
    entropies = np.array(entropies)
    
    min_ent = entropies.min()
    max_ent = entropies.max()
    
    if max_ent - min_ent == 0:
        normalized_entropies = np.zeros_like(entropies)
    else:
        normalized_entropies = (entropies - min_ent) / (max_ent - min_ent)
    average_normalized_entropy = normalized_entropies.mean()
    return average_normalized_entropy


def run_S_Branch(
    path_file,
    K=3,
    device='cuda',
    bbox_extension=4,
    sort_descending=True
):
    class S_Branch:
        def __init__(self, K=3, device='cuda'):
            self.K = K
            self.N = 5 * K
            self.device = device
            self.sigma = 1e-5
            self.prepare_kernels()

        def prepare_kernels(self):
            K, N = self.K, self.N
            op_in = torch.ones((1, 1, K, K), device=self.device)
            op_in_m = op_in / op_in.sum()
            self.op_in_m = op_in_m
            op_mid = torch.zeros((1, 1, N, N), device=self.device)
            op_mid[:, :, K:4*K, K:4*K] = 1
            op_mid[:, :, 2*K:3*K, 2*K:3*K] = 0
            op_mid_m = op_mid / op_mid.sum()
            self.op_mid_m = op_mid_m
            op_out = torch.ones((1, 1, N, N), device=self.device)
            op_out[:, :, K:4*K, K:4*K] = 0
            op_out_m = op_out / op_out.sum()
            self.op_out_m = op_out_m
            self.pad_in = (self.K // 2, self.K // 2, self.K // 2, self.K // 2)
            self.pad_mid = (self.N // 2, self.N // 2, self.N // 2, self.N // 2)
            self.pad_out = (self.N // 2, self.N // 2, self.N // 2, self.N // 2)

        def run_S_Branch(self, img):
            if len(img.shape) != 4:
                raise ValueError(f"Expected input tensor to be 4D (1, 1, H, W), but got {img.shape}")
            img_padded_in = F.pad(img, self.pad_in, mode='replicate')
            img_mean_in = F.conv2d(img_padded_in, self.op_in_m)
            img_padded_mid = F.pad(img, self.pad_mid, mode='replicate')
            img_mean_mid = F.conv2d(img_padded_mid, self.op_mid_m)
            img_padded_out = F.pad(img, self.pad_out, mode='replicate')
            img_mean_out = F.conv2d(img_padded_out, self.op_out_m)
            out1 = (img_mean_in - img_mean_mid) + (img_mean_in - img_mean_out)
            out1 = torch.clamp(out1, min=0)
            op_mid_variance = torch.zeros((1, 1, self.N, self.N), device=self.device)
            op_mid_variance[:, :, self.K:4*self.K, self.K:4*self.K] = 1
            op_mid_variance[:, :, 2*self.K:3*self.K, 2*self.K:3*self.K] = 1  
            op_mid_variance = op_mid_variance / op_mid_variance.sum()

            img_mean_mid_var = F.conv2d(F.pad(img, self.pad_mid, mode='replicate'), op_mid_variance)
            img_mean_sq_mid = F.conv2d(F.pad(img ** 2, self.pad_mid, mode='replicate'), op_mid_variance)
            img_std_mid = torch.sqrt(torch.clamp(img_mean_sq_mid - img_mean_mid_var ** 2, min=0))

            img_std_out = F.conv2d(F.pad(img ** 2, self.pad_out, mode='replicate'), self.op_out_m)
            img_std_out = torch.sqrt(torch.clamp(img_std_out - (F.conv2d(F.pad(img, self.pad_out, mode='replicate'), self.op_out_m)) ** 2, min=0))

            out2 = (img_std_mid / (img_std_out + self.sigma)) * img_std_mid - img_std_mid
            out2 = torch.clamp(out2, min=0)
            
            out = out1 * out2
            return out

    def mask_to_bboxes(mask, max_bboxes=1, bbox_extension=4, sort_descending=True):

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None  

        bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append([x, y, x + w + bbox_extension, y + h + bbox_extension]) 

        if bboxes:
            bboxes = sorted(
                bboxes,
                key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
                reverse=sort_descending
            )
            if max_bboxes is not None:
                bboxes = bboxes[:max_bboxes]

        return bboxes if bboxes else None

    S_G = S_Branch(K=K, device=device)

    image_paths = []
    with open(path_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                image_path = line.split(' ')[0]
                image_paths.append(image_path)

    S_G_output = OrderedDict()

    for img_path in tqdm(image_paths, desc="Processing images with S_Branch"):
        if not os.path.exists(img_path):
            print(f"Warning: Image path does not exist: {img_path}. Skipping this image.")
            S_G_output[img_path] = []
            continue

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Unable to read image: {img_path}. Skipping this image.")
            S_G_output[img_path] = []
            continue

        height, width = image.shape[:2]
        img_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0) / 255.0 
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            out = S_G.run_S_Branch(img_tensor)

        out_np = out.squeeze().cpu().numpy()
        out_np = np.nan_to_num(out_np)
        out_norm = (out_np - out_np.min()) / (out_np.max() - out_np.min() + 1e-8)
        threshold = 0.5
        out_binary = np.zeros_like(out_norm)
        out_binary[out_norm >= threshold] = 1

        extracted_bboxes = mask_to_bboxes(
            out_binary,
            max_bboxes=1,
            bbox_extension=bbox_extension,
            sort_descending=sort_descending
        )

        if extracted_bboxes:
            bboxes = [tuple(map(int, bbox)) for bbox in extracted_bboxes]
            S_G_output[img_path] = bboxes
        else:
            S_G_output[img_path] = []

    return S_G_output




class TrackedObject:
    def __init__(self, bbox, object_id, frame_number):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.object_id = object_id
        self.last_seen = frame_number
        self.num_consecutive_frames_seen = 1
        self.frames_lost = 0 

class Tracker:
    def __init__(self, max_lost=5, min_hits=3, iou_threshold=0.3):
        self.next_object_id = 1
        self.tracked_objects = {}
        self.max_lost = max_lost 
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

    def update(self, detections, frame_number):
        if len(self.tracked_objects) == 0:
            for bbox in detections:
                tracked_obj = TrackedObject(bbox, self.next_object_id, frame_number)
                self.tracked_objects[self.next_object_id] = tracked_obj
                self.next_object_id += 1
            return
        matches, unmatched_detections, unmatched_tracked_objects = self.assign_detections_to_trackers(detections)

        for t_id, det_idx in matches.items():
            bbox = detections[det_idx]
            tracked_obj = self.tracked_objects[t_id]
            tracked_obj.bbox = bbox
            tracked_obj.last_seen = frame_number
            tracked_obj.num_consecutive_frames_seen += 1
            tracked_obj.frames_lost = 0

        for det_idx in unmatched_detections:
            bbox = detections[det_idx]
            tracked_obj = TrackedObject(bbox, self.next_object_id, frame_number)
            self.tracked_objects[self.next_object_id] = tracked_obj
            self.next_object_id += 1
        to_delete = []
        for t_id in unmatched_tracked_objects:
            tracked_obj = self.tracked_objects[t_id]
            tracked_obj.frames_lost += 1
            if tracked_obj.frames_lost > self.max_lost:
                to_delete.append(t_id)
        for t_id in to_delete:
            del self.tracked_objects[t_id]

    def assign_detections_to_trackers(self, detections):
        iou_matrix = np.zeros((len(self.tracked_objects), len(detections)), dtype=np.float32)
        tracked_ids = list(self.tracked_objects.keys())
        for t_idx, t_id in enumerate(tracked_ids):
            t_bbox = self.tracked_objects[t_id].bbox
            for d_idx, det_bbox in enumerate(detections):
                iou = compute_iou(t_bbox, det_bbox)
                iou_matrix[t_idx, d_idx] = iou

        matched_indices = []
        if iou_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array(list(zip(row_ind, col_ind)))

        matches = {}
        unmatched_detections = list(range(len(detections)))
        unmatched_tracked_objects = tracked_ids.copy()

        for t_idx, d_idx in matched_indices:
            if iou_matrix[t_idx, d_idx] >= self.iou_threshold:
                t_id = tracked_ids[t_idx]
                matches[t_id] = d_idx
                unmatched_detections.remove(d_idx)
                unmatched_tracked_objects.remove(t_id)
            else:
                continue  
        return matches, unmatched_detections, unmatched_tracked_objects


        
def generate_pseudo_labels(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    frame_data = defaultdict(list)

    sequences = defaultdict(list)
    for line in lines:
        parts = line.strip().split()
        img_path = parts[0]
        frame_number = int(os.path.splitext(os.path.basename(img_path))[0])
        sequence_number = int(os.path.basename(os.path.dirname(img_path)))
        sequences[sequence_number].append((frame_number, img_path))

    orb = cv2.ORB_create(nfeatures=5000) 
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    total_frames = sum(len(frames) for frames in sequences.values())

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        for sequence_number in sequences.keys():
            frames = sequences[sequence_number]
            frames.sort(key=lambda x: x[0]) 
            previous_frame = None
            previous_kp = None
            previous_des = None
            tracker = Tracker(max_lost=5, min_hits=2, iou_threshold=0.3)

            for frame_number, img_path in frames:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    frame_data[sequence_number].append((frame_number, []))
                    pbar.update(1)
                    continue
                img_enhanced = cv2.equalizeHist(img)
                if previous_frame is not None:
                    kp, des = orb.detectAndCompute(img_enhanced, None)
                    if des is None or previous_des is None:
                        frame_diff = cv2.absdiff(img_enhanced, previous_frame)
                    else:
                        matches = bf.match(previous_des, des)
                        matches = sorted(matches, key=lambda x: x.distance)
                        if len(matches) >= 4:
                            src_pts = np.float32([previous_kp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
                            dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
                            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                            if M is not None:
                                aligned_img = cv2.warpPerspective(img_enhanced, M, (img_enhanced.shape[1], img_enhanced.shape[0]))
                                frame_diff = cv2.absdiff(aligned_img, previous_frame)
                            else:
                                frame_diff = cv2.absdiff(img_enhanced, previous_frame)
                        else:
                            frame_diff = cv2.absdiff(img_enhanced, previous_frame)
                else:
                    frame_data[sequence_number].append((frame_number, []))
                    previous_frame = img_enhanced
                    previous_kp, previous_des = orb.detectAndCompute(previous_frame, None)
                    pbar.update(1)
                    continue
                previous_frame = img_enhanced
                previous_kp = kp
                previous_des = des

                _, diff_thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
                diff_thresh = cv2.dilate(diff_thresh, kernel, iterations=2)
                contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                targets = []

                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if 4 < w < 20 and 4 < h < 20: 
                        targets.append((x, y, x + w, y + h))
                targets = sorted(targets, key=lambda box: (box[2]-box[0])*(box[3]-box[1]), reverse=True)[:2]
                tracker.update(targets, frame_number)
                valid_tracked_objects = []
                for t_id, tracked_obj in tracker.tracked_objects.items():
                    if tracked_obj.num_consecutive_frames_seen >= tracker.min_hits and tracked_obj.frames_lost == 0:
                        valid_tracked_objects.append(tracked_obj.bbox)
                frame_data[sequence_number].append((frame_number, valid_tracked_objects))
                pbar.update(1)
        for sequence_number in frame_data:
            frame_data[sequence_number].sort(key=lambda x: x[0])
        return frame_data



def post_process_pseudo_labels(frame_data):
    class Track:
        def __init__(self, track_id, initial_position):
            self.track_id = track_id
            self.positions = deque([initial_position], maxlen=250) 
            self.velocities = deque(maxlen=10) 
            self.last_known_position = initial_position
            self.last_known_velocity = None
            self.frames_since_last_seen = 0
            self.num_consecutive_frames_seen = 1

        def update(self, new_position):
            self.frames_since_last_seen = 0
            self.num_consecutive_frames_seen += 1
            if len(self.positions) > 0:
                velocity = tuple(self.last_known_position[i] - new_position[i] for i in range(4))
                self.velocities.appendleft(velocity)
                self.last_known_velocity = tuple(
                    sum(v[i] for v in self.velocities) / len(self.velocities) for i in range(4)
                )
            self.positions.appendleft(new_position)
            self.last_known_position = new_position

        def predict_position(self):
            if self.last_known_velocity:
                predicted_position = tuple(
                    int(self.last_known_position[i] - self.last_known_velocity[i]) for i in range(4)
                )
                return predicted_position
            else:
                return self.last_known_position

    def compute_iou(box1, box2):
        x1_min, y1_min, x1_max, y1_max = map(int, box1)
        x2_min, y2_min, x2_max, y2_max = map(int, box2)

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
            return 0.0 

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)

        iou = inter_area / float(area1 + area2 - inter_area)
        return iou

    def compute_center_distance(box1, box2):
        x1_min, y1_min, x1_max, y1_max = map(int, box1)
        x2_min, y2_min, x2_max, y2_max = map(int, box2)

        center1_x = (x1_min + x1_max) / 2
        center1_y = (y1_min + y1_max) / 2
        center2_x = (x2_min + x2_max) / 2
        center2_y = (y2_min + y2_max) / 2

        distance = np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)
        return distance

    processed_frame_data = defaultdict(list)
    max_frames_to_keep = 6
    distance_threshold = 80
    iou_threshold = 0.001 
    merge_iou_threshold = 0.01

    for sequence, frames in frame_data.items():
        tracks = []
        track_id_counter = 0
        frames_reversed = frames[::-1] 
        frame_outputs = [] 

        for i, (frame_number, targets) in enumerate(frames_reversed):
            predicted_positions = []
            for track in tracks:
                predicted_position = track.predict_position()
                predicted_positions.append((track, predicted_position))
            detection_centers = []
            for target in targets:
                x1, y1, x2, y2 = target
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                detection_centers.append(((x_center, y_center), target))

            assignments = []
            if predicted_positions and detection_centers:
                cost_matrix = []
                for track, predicted_position in predicted_positions:
                    track_x1, track_y1, track_x2, track_y2 = predicted_position
                    track_x_center = (track_x1 + track_x2) / 2
                    track_y_center = (track_y1 + track_y2) / 2
                    costs = []
                    for (det_x_center, det_y_center), _ in detection_centers:
                        distance = np.hypot(track_x_center - det_x_center, track_y_center - det_y_center)
                        costs.append(distance)
                    cost_matrix.append(costs)

                cost_matrix = np.array(cost_matrix)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                assigned_tracks = []
                assigned_detections = []

                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] < distance_threshold:
                        track = predicted_positions[r][0]
                        (_, _), detection = detection_centers[c]
                        assignments.append((track, detection))
                        assigned_tracks.append(track)
                        assigned_detections.append(c)
                        track.update(detection)
                    else:
                        track = predicted_positions[r][0]
                        track.frames_since_last_seen += 1
                unassigned_tracks = [track for track in tracks if track not in assigned_tracks]
                for track in unassigned_tracks:
                    track.frames_since_last_seen += 1
                unassigned_detections = [detection_centers[idx] for idx in range(len(detection_centers)) if idx not in assigned_detections]
                for (_, _), detection in unassigned_detections:
                    track_id_counter += 1
                    new_track = Track(track_id_counter, detection)
                    tracks.append(new_track)
                    assignments.append((new_track, detection))
            else:
                for (_, _), detection in detection_centers:
                    track_id_counter += 1
                    new_track = Track(track_id_counter, detection)
                    tracks.append(new_track)
                    assignments.append((new_track, detection))

                for track in tracks:
                    if track not in [a[0] for a in assignments]:
                        track.frames_since_last_seen += 1

            tracks = [track for track in tracks if track.frames_since_last_seen <= max_frames_to_keep]

            merged_tracks = []
            for track in tracks:
                merged = False
                for m_track in merged_tracks:
                    iou = compute_iou(track.last_known_position, m_track.last_known_position)
                    if track.last_known_velocity and m_track.last_known_velocity:
                        velocity_diff = np.linalg.norm(
                            np.array(track.last_known_velocity) - np.array(m_track.last_known_velocity))
                    else:
                        velocity_diff = float('inf')
                    if iou > merge_iou_threshold and velocity_diff < distance_threshold:
                        m_track.positions.extendleft(reversed(track.positions))
                        m_track.velocities.extendleft(reversed(track.velocities))
                        m_track.last_known_position = track.last_known_position
                        if m_track.velocities:
                            m_track.last_known_velocity = tuple(
                                sum(v[i] for v in m_track.velocities) / len(m_track.velocities) for i in range(4)
                            )
                        merged = True
                        break
                if not merged:
                    merged_tracks.append(track)
            tracks = merged_tracks
            detections_in_frame = []
            for track, detection in assignments:
                if track in tracks:
                    detections_in_frame.append((track, detection))
            for track in tracks:
                if track.frames_since_last_seen > 0 and track not in [a[0] for a in assignments]:
                    predicted_position = track.predict_position()
                    x1, y1, x2, y2 = predicted_position
                    detections_in_frame.append((track, (x1, y1, x2, y2)))
            filtered_detections = []
            for idx, (track1, det1) in enumerate(detections_in_frame):
                keep_current = True
                to_remove = []
                for j, (track2, det2) in enumerate(filtered_detections):
                    iou = compute_iou(det1, det2)
                    if iou > iou_threshold:
                        if len(track1.positions) > len(track2.positions):
                            to_remove.append(j)
                        else:
                            keep_current = False
                            break
                for j in sorted(to_remove, reverse=True):
                    del filtered_detections[j]
                if keep_current:
                    filtered_detections.append((track1, det1))
            filtered_detections.sort(key=lambda x: x[0].track_id)
            if filtered_detections:
                final_boxes = [(int(x1), int(y1), int(x2), int(y2)) for _, (x1, y1, x2, y2) in filtered_detections]
            else:
                final_boxes = []

            frame_outputs.append((frame_number, final_boxes))
        for frame_number, boxes in frame_outputs[::-1]:
            processed_frame_data[sequence].append((frame_number, boxes))
    return processed_frame_data
    
    

def compute_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = map(int, box1[:4])
    x2_min, y2_min, x2_max, y2_max = map(int, box2[:4])

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        return 0.0  

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    iou = inter_area / float(area1 + area2 - inter_area)
    return iou

def compute_center_distance(box1, box2):
    x1_min, y1_min, x1_max, y1_max = map(int, box1)
    x2_min, y2_min, x2_max, y2_max = map(int, box2)

    center1_x = (x1_min + x1_max) / 2
    center1_y = (y1_min + y1_max) / 2
    center2_x = (x2_min + x2_max) / 2
    center2_y = (y2_min + y2_max) / 2

    distance = np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)
    return distance



def filter_overlapping_boxes(frame_data, input_file, iou_threshold=0.01, distance_threshold=15):
    with open(input_file, 'r') as f:
        input_lines = [line.strip() for line in f if line.strip()]

    bbox_frequency = defaultdict(int)
    for seq_num, frames in frame_data.items():
        for frame_num, boxes in frames:
            for box in boxes:
                box_str = f"{box[0]},{box[1]},{box[2]},{box[3]},0"
                bbox_frequency[box_str] += 1

    output_dict = {}

    for line in input_lines:
        parts = line.split()
        img_path = parts[0]

        try:
            path_parts = img_path.split('/')
            sequence_number_str = path_parts[-2]
            frame_filename = path_parts[-1]
            frame_number_str = frame_filename.split('.')[0]
            sequence_number = int(sequence_number_str)
            frame_number = int(frame_number_str)
        except (IndexError, ValueError) as e:
            output_dict[img_path] = []
            continue
        boxes = []
        if sequence_number in frame_data:
            for frame in frame_data[sequence_number]:
                if frame[0] == frame_number:
                    boxes = frame[1]
                    break

        if not boxes:
            output_dict[img_path] = []
            continue

        img = cv2.imread(img_path)
        if img is None:
            output_dict[img_path] = []
            continue
        height, width = img.shape[:2]

        bbox_list = []
        for box in boxes:
            if len(box) < 4:
                continue
            x1, y1, x2, y2 = box[:4]
            if x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1:
                continue 
            box_str = f"{x1},{y1},{x2},{y2},0"
            bbox_list.append((box_str, (x1, y1, x2, y2, 0))) 

        keep_boxes = []
        used_indices = set()
        for i, (box1_str, box1_coords) in enumerate(bbox_list):
            if i in used_indices:
                continue
            overlaps = [(i, box1_str, box1_coords)]
            for j, (box2_str, box2_coords) in enumerate(bbox_list):
                if j <= i or j in used_indices:
                    continue
                iou = compute_iou(box1_coords[:4], box2_coords[:4])
                distance = compute_center_distance(box1_coords[:4], box2_coords[:4])
                if iou > iou_threshold or distance < distance_threshold:
                    overlaps.append((j, box2_str, box2_coords))
            if overlaps:
                overlaps.sort(key=lambda x: bbox_frequency.get(x[1], 0), reverse=True)
                best_box = overlaps[0][2] 
                keep_boxes.append(best_box)
                for overlap in overlaps:
                    used_indices.add(overlap[0])
            else:
                keep_boxes.append(box1_coords)
                used_indices.add(i)
        output_dict[img_path] = keep_boxes

    return output_dict




def homogeneous_motion_hybridizing(input_file, output_dir, augmented_input_file, num_targets_per_sequence):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    with open(input_file, 'r') as f:
        lines = f.readlines()

    frame_data = defaultdict(list)
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        img_path = parts[0]
        boxes = parts[1:]
        try:
            frame_number = int(os.path.splitext(os.path.basename(img_path))[0])
            sequence_number = int(os.path.basename(os.path.dirname(img_path)))
        except ValueError:
            continue
        frame_data[sequence_number].append({
            'frame_number': frame_number,
            'img_path': img_path,
            'boxes': boxes
        })

    for seq in frame_data.values():
        seq.sort(key=lambda x: x['frame_number'])
    sequence_tracks = {}
    for sequence_number, frames in frame_data.items():
        tracks = {}
        for frame in frames:
            frame_number = frame['frame_number']
            img_path = frame['img_path']
            boxes = frame['boxes']
            for box_str in boxes:
                box_parts = box_str.split(',')
                if len(box_parts) >= 5:
                    try:
                        x1, y1, x2, y2, track_id = box_parts[:5]
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        track_id = int(track_id)
                    except ValueError:
                        continue
                else:
                    continue
                if track_id not in tracks:
                    tracks[track_id] = []
                tracks[track_id].append({
                    'frame_number': frame_number,
                    'img_path': img_path,
                    'bbox': [x1, y1, x2, y2]
                })
        sequence_tracks[sequence_number] = tracks

    for sequence_number, frames in frame_data.items():
        for frame in frames:
            src_frame_path = frame['img_path']
            frame_number = frame['frame_number']
            dst_seq_dir = os.path.join(output_dir, str(sequence_number))
            if not os.path.exists(dst_seq_dir):
                os.makedirs(dst_seq_dir)
            dst_frame_path = os.path.join(dst_seq_dir, f"{frame_number}.bmp")
            if not os.path.exists(dst_frame_path):
                shutil.copyfile(src_frame_path, dst_frame_path)

    seq_numbers = list(frame_data.keys())
    frame_targets = defaultdict(lambda: defaultdict(list))

    def remove_background_with_grabcut(img):

        if img is None or img.size == 0:
            return None

        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)

        h, w = img.shape[:2]
        if h < 2 or w < 2:
            alpha = np.ones((h, w), dtype=np.uint8)*255
            return np.dstack((img, alpha))
        rect = (1, 1, w-2, h-2)

        try:
            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        except:
            alpha = np.ones((h, w), dtype=np.uint8)*255
            return np.dstack((img, alpha))

        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        fg = img * mask2[:,:,np.newaxis]
        alpha = (mask2 * 255).astype(np.uint8)
        fg = np.dstack((fg, alpha))
        return fg

    def get_tight_object(img):
        if img is None or img.size == 0:
            return None
        if img.shape[2] < 4:
            alpha_ch = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)*255
            img = np.dstack((img, alpha_ch))
        else:
            alpha_ch = img[:,:,3]

        ys, xs = np.where(alpha_ch > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()

        return img[min_y:max_y+1, min_x:max_x+1]

    def try_get_base_target_object(target_track):
        for item in target_track:
            img = cv2.imread(item['img_path'], cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            x1, y1, x2, y2 = item['bbox']
            h, w = img.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            target_crop = img[y1:y2, x1:x2]
            if target_crop.size == 0:
                continue
            segmented = remove_background_with_grabcut(target_crop)
            if segmented is None or segmented.size == 0:
                continue
            tight_obj = get_tight_object(segmented)
            if tight_obj is not None and tight_obj.size > 0:
                if tight_obj.shape[2] == 4:
                    alpha_ch = tight_obj[:, :, 3]
                    foreground_pixels = np.sum(alpha_ch > 0)
                    if foreground_pixels < 8:
                        continue
                    return tight_obj

        return None

    for seq_num in seq_numbers:
        current_frames = frame_data[seq_num]
        num_targets_to_add = num_targets_per_sequence

        for _ in range(num_targets_to_add):
            target_seq_num = random.choice(seq_numbers)
            if target_seq_num == seq_num and len(seq_numbers) > 1:
                target_seq_num = random.choice([num for num in seq_numbers if num != seq_num])
            target_tracks = sequence_tracks.get(target_seq_num, {})
            if not target_tracks:
                continue
            target_track_id = random.choice(list(target_tracks.keys()))
            target_track = target_tracks[target_track_id]

            motion_seq_num = random.choice(seq_numbers)
            if motion_seq_num == seq_num and len(seq_numbers) > 1:
                motion_seq_num = random.choice([num for num in seq_numbers if num != seq_num])
            motion_tracks = sequence_tracks.get(motion_seq_num, {})
            if not motion_tracks:
                continue
            motion_track_id = random.choice(list(motion_tracks.keys()))
            motion_track = motion_tracks[motion_track_id]

            min_length = min(len(current_frames), len(target_track), len(motion_track))
            if min_length < 3:
                continue
            target_track = target_track[:min_length]
            motion_track = motion_track[:min_length]
            current_frames_subset = current_frames[:min_length]

            base_target_object = try_get_base_target_object(target_track)
            if base_target_object is None:
                continue

            th, tw = base_target_object.shape[:2]
            area = tw * th
            is_small_target = (area < 10000)

            existing_ids = list(sequence_tracks[seq_num].keys())
            new_track_id = max(existing_ids) + 1 if existing_ids else 0
            sequence_tracks[seq_num][new_track_id] = []

            for idx, frame_item in enumerate(current_frames_subset):
                frame_number = frame_item['frame_number']
                frame_path = os.path.join(output_dir, str(seq_num), f"{frame_number}.bmp")
                frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
                if frame is None:
                    continue

                transformed_object = base_target_object.copy()

                if is_small_target:
                    angle = random.uniform(-15, 15)
                    scale = random.uniform(1.00, 1.00)
                    flip = random.choice([True, False])
                    center = (transformed_object.shape[1] / 2, transformed_object.shape[0] / 2)
                    M = cv2.getRotationMatrix2D(center, angle, scale)
                    transformed_object = cv2.warpAffine(transformed_object, M, 
                        (transformed_object.shape[1], transformed_object.shape[0]),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                    if flip:
                        transformed_object = cv2.flip(transformed_object, 1)

                    if transformed_object.ndim == 2 or transformed_object.shape[2] == 3:
                        alpha = np.ones((transformed_object.shape[0], transformed_object.shape[1]), dtype=np.uint8)*255
                        transformed_object = np.dstack((transformed_object, alpha))
                    transformed_object = get_tight_object(transformed_object)
                    if transformed_object is None or transformed_object.size == 0:
                        continue

                    alpha_ch = transformed_object[:, :, 3]
                    if np.sum(alpha_ch > 0) < 8:
                        continue

                motion_bbox = motion_track[idx]['bbox']
                mx1, my1, mx2, my2 = motion_bbox

                motion_offset_x = random.randint(-1, 1)
                motion_offset_y = random.randint(-1, 1)
                mx1 += motion_offset_x
                my1 += motion_offset_y

                th, tw = transformed_object.shape[:2]
                overlay_x1 = mx1
                overlay_y1 = my1
                overlay_x2 = overlay_x1 + tw
                overlay_y2 = overlay_y1 + th

                h_frame, w_frame = frame.shape[:2]
                overlay_x1_clamped = max(0, min(int(overlay_x1), w_frame - 1))
                overlay_y1_clamped = max(0, min(int(overlay_y1), h_frame - 1))
                overlay_x2_clamped = max(0, min(int(overlay_x2), w_frame))
                overlay_y2_clamped = max(0, min(int(overlay_y2), h_frame))

                dx1 = overlay_x1_clamped - overlay_x1
                dy1 = overlay_y1_clamped - overlay_y1
                dx2 = overlay_x2 - overlay_x2_clamped
                dy2 = overlay_y2 - overlay_y2_clamped

                dx1 = int(dx1)
                dy1 = int(dy1)
                dx2 = int(dx2)
                dy2 = int(dy2)

                target_crop = transformed_object[dy1:th - dy2, dx1:tw - dx2]
                if target_crop.size == 0:
                    continue

                if target_crop.shape[2] == 3:
                    alpha_s = np.ones((target_crop.shape[0], target_crop.shape[1], 1), dtype=np.float32)
                    target_rgb = target_crop
                else:
                    alpha_s = target_crop[:, :, 3] / 255.0
                    alpha_s = alpha_s[:, :, np.newaxis]
                    target_rgb = target_crop[:, :, :3]

                roi = frame[overlay_y1_clamped:overlay_y2_clamped, overlay_x1_clamped:overlay_x2_clamped]
                if roi.shape[:2] != target_rgb.shape[:2]:
                    continue

                frame[overlay_y1_clamped:overlay_y2_clamped, overlay_x1_clamped:overlay_x2_clamped] = (
                    alpha_s * target_rgb + (1 - alpha_s) * roi
                ).astype(np.uint8)

                aug_seq_dir = os.path.join(output_dir, str(seq_num))
                if not os.path.exists(aug_seq_dir):
                    os.makedirs(aug_seq_dir)
                aug_frame_path = os.path.join(aug_seq_dir, f"{frame_number}.bmp")
                cv2.imwrite(aug_frame_path, frame)

                final_bbox = (overlay_x1_clamped, overlay_y1_clamped, overlay_x2_clamped + 1, overlay_y2_clamped + 1)
                sequence_tracks[seq_num][new_track_id].append({
                    'frame_number': frame_number,
                    'img_path': aug_frame_path,
                    'bbox': final_bbox
                })
                frame_targets[seq_num][frame_number].append((final_bbox, new_track_id))

    augmented_lines = []
    for sequence_number, frames in frame_data.items():
        for frame in frames:
            frame_number = frame['frame_number']
            aug_frame_path = os.path.join(output_dir, str(sequence_number), f"{frame_number}.bmp")
            original_boxes = frame['boxes']
            augmented_boxes = frame_targets[sequence_number].get(frame_number, [])
            all_boxes = original_boxes.copy()
            for bbox, track_id in augmented_boxes:
                x1, y1, x2, y2 = map(int, bbox)
                all_boxes.append(f"{x1},{y1},{x2},{y2},0")
            if all_boxes:
                boxes_str = ' '.join(all_boxes)
                augmented_lines.append(f"{aug_frame_path} {boxes_str}\n")
            else:
                augmented_lines.append(f"{aug_frame_path}\n")

    with open(augmented_input_file, 'w') as f_out:
        f_out.writelines(augmented_lines)



if __name__ == "__main__":
    generation()