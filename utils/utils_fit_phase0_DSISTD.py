import os
import torch
from tqdm import tqdm
from utils.utils import get_lr
import cv2
import numpy as np
from collections import defaultdict
import math
from collections import deque
from collections import Counter
import re
from scipy.optimize import linear_sum_assignment
from predict import Pred_vid
from test import get_history_imgs
from PIL import Image
import shutil
import random



def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss        = 0

    ##################################
    input_file = '/home/public/DSISTD/train_pseudo_labels.txt'
    candidate_file = '/home/public/DSISTD/candidate_pseudo_labels.txt'
    path_file = '/home/public/DSISTD/train.txt'
    output_dir = '/home/public/DSISTD_AUG'
    collected_labels = {}
    ##################################

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]

        optimizer.zero_grad()
        if not fp16:
            outputs = model_train(images)
            loss_value = yolo_loss(outputs, targets) 
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(images)
                loss_value = yolo_loss(outputs, targets)

            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)

        loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1),
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')

        loss_history.append_loss(epoch + 1, loss / epoch_step, 0)

        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f ' % (loss / epoch_step))

        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f.pth" % (epoch + 1, loss / epoch_step)))

        # Adjust the condition to save the best model based on training loss
        if len(loss_history.losses) <= 1 or (loss / epoch_step) <= min(loss_history.losses):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))

    if (epoch + 1) >=5:
        print('-------Pseudo-label collection-------')
        detector = Pred_vid(
            model_path=os.path.join(save_dir, "last_epoch_weights.pth"),
            classes_path='model_data/classes.txt',
            input_shape=[512, 512],
            phi='s',
            confidence=0.3,
            nms_iou=0.5,
            letterbox_image=True,
            cuda=True
        )
        detector.net.eval() 

        collected_image_paths = []
        collected_predictions = []

        with open(path_file, 'r') as f_train:
            train_lines = f_train.readlines()

        total_images = len(train_lines)
        pbar_eval = tqdm(total=total_images, desc="Collecting", mininterval=0.3)
        with torch.no_grad():
            for line in train_lines:
                img_path = line.strip().split()[0]
                try:
                    img_list_paths = get_history_imgs(img_path)
                    images = [Image.open(item) for item in img_list_paths]
                except Exception as e:
                    print(f"Unable to open image {img_path}, error: {e}")
                    pbar_eval.update(1)
                    continue

                _, boxes = detector.detect_image(images, crop=False, count=False)

                all_boxes = boxes.copy()
                all_boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]
                
                collected_predictions.append(all_boxes)
                collected_image_paths.append(img_path)  

                pbar_eval.update(1)
        pbar_eval.close()

        collect_pseudo_labels(collected_image_paths, collected_predictions, candidate_file, input_file)
        print('-------Candidate labels are collected-------')
        update_pseudo_labels(input_file, candidate_file, epoch+1, specified_epoch=5, iou_threshold=0.1)
        print('-------Labels update completed-------')

        filter_targets(input_file)
        
        print("Augmenting pseudo-labels...")
        augment_pseudo_labels_with_motion(input_file,output_dir,input_file,2)



def collect_pseudo_labels(image_paths, best_predictions, candidate_file, input_file):
    collected_labels = {}
    for batch_idx, predictions in enumerate(best_predictions):
        img_path_with_label = image_paths[batch_idx]  
        img_path = img_path_with_label.split()[0] 

        if len(predictions) > 0:
            for prediction in predictions:
                if len(prediction) >= 4:
                    x1 = int(prediction[0])
                    y1 = int(prediction[1])
                    x2 = int(prediction[2])
                    y2 = int(prediction[3])
                    target_str = f"{x1},{y1},{x2},{y2},0"

                    if img_path in collected_labels:
                        collected_labels[img_path].append(target_str)
                    else:
                        collected_labels[img_path] = [target_str]
                else:
                    print(f"Warning: The format of the prediction box is incorrect and will be skipped. Prediction: {prediction}")
        else:
            if img_path not in collected_labels:
                collected_labels[img_path] = []

    existing_candidates = {}

    if os.path.exists(candidate_file):
        with open(candidate_file, "r") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    existing_img_path, labels_str = parts
                    labels = labels_str.strip().split()
                    existing_candidates[existing_img_path] = labels
                elif len(parts) == 1:
                    existing_img_path = parts[0]
                    existing_candidates[existing_img_path] = []

    for img_path, labels in collected_labels.items():
        if img_path in existing_candidates:
            existing_candidates[img_path].extend(labels)
        else:
            existing_candidates[img_path] = labels

    for img_path_with_label in image_paths:
        img_path = img_path_with_label.split()[0]
        if img_path not in existing_candidates:
            existing_candidates[img_path] = []
            
    if os.path.exists(input_file):
        with open(input_file, "r") as f:
            input_image_paths = [line.strip().split()[0] for line in f if line.strip()]
    else:
        input_image_paths = []

    with open(candidate_file, "w") as f:
        for img_path in input_image_paths:
            labels = existing_candidates.get(img_path, [])
            if labels:
                labels_with_count = ' '.join(labels)
                f.write(f"{img_path} {labels_with_count}\n")
            else:
                f.write(f"{img_path}\n")



def update_pseudo_labels(input_file, candidate_file, epoch, specified_epoch=1, iou_threshold=0.3):
    def average_boxes(boxes):
        """Compute the average bounding box from a list of boxes."""
        x1s = [box[0] for box in boxes]
        y1s = [box[1] for box in boxes]
        x2s = [box[2] for box in boxes]
        y2s = [box[3] for box in boxes]
        avg_x1 = int(round(sum(x1s) / len(x1s)))
        avg_y1 = int(round(sum(y1s) / len(y1s)))
        avg_x2 = int(round(sum(x2s) / len(x2s)))
        avg_y2 = int(round(sum(y2s) / len(y2s)))
        return [avg_x1, avg_y1, avg_x2, avg_y2]

    candidate_labels = {}

    if not os.path.exists(candidate_file):
        print(f"Candidate file {candidate_file} not found.")
        return

    with open(candidate_file, 'r') as f_candidate:
        for line in f_candidate:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                img_path, labels_str = parts
                labels = labels_str.strip().split()
                labels_with_counts = []
                for label in labels:
                    label_parts = label.split(',')
                    if len(label_parts) == 5:
                        x1, y1, x2, y2, count = label_parts
                        labels_with_counts.append([int(x1), int(y1), int(x2), int(y2), int(count)])
                    else:
                        print(f"Warning: Label format incorrect in candidate_file: {label}")
                candidate_labels[img_path] = labels_with_counts
            elif len(parts) == 1:
                img_path = parts[0]
                candidate_labels[img_path] = []

    # Increment counts and remove targets with count >=6
    for img_path in candidate_labels:
        labels = candidate_labels[img_path]
        new_labels = []
        for label in labels:
            x1, y1, x2, y2, count = label
            count = int(count) + 1  # Increment count by 1
            if count < 6:
                new_labels.append([x1, y1, x2, y2, count])
            # Else, if count >=6, we discard this label
        candidate_labels[img_path] = new_labels

    # Read input_labels
    input_labels = {}
    try:
        with open(input_file, 'r') as f_input:
            for line in f_input:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    img_path, labels_str = parts
                    labels = labels_str.strip().split()
                    labels_list = []
                    for label in labels:
                        label_parts = label.split(',')
                        if len(label_parts) >= 4:
                            x1, y1, x2, y2 = label_parts[:4]
                            labels_list.append([int(x1), int(y1), int(x2), int(y2)])
                    input_labels[img_path] = labels_list
                elif len(parts) == 1:
                    img_path = parts[0]
                    input_labels[img_path] = []
    except FileNotFoundError:
        print(f"input_file {input_file} not found, creating new input_file.")
    except Exception as e:
        print(f"Error reading input_file {input_file}: {e}")
        return

    if epoch == specified_epoch:
        input_labels = {}
        for img_path in candidate_labels:
            labels = candidate_labels[img_path]
            labels_no_count = []
            for label in labels:
                x1, y1, x2, y2, count = label
                labels_no_count.append([x1, y1, x2, y2])
            input_labels[img_path] = labels_no_count

        for img_path in input_labels:
            labels = input_labels[img_path] 
            num_labels = len(labels)
            if num_labels == 0:
                continue

            adjacency = [[] for _ in range(num_labels)]
            for i in range(num_labels):
                for j in range(i + 1, num_labels):
                    box1 = labels[i]
                    box2 = labels[j]
                    iou = compute_iou(box1, box2)
                    if iou > iou_threshold:
                        adjacency[i].append(j)
                        adjacency[j].append(i)

            visited = [False] * num_labels
            clusters = []
            for i in range(num_labels):
                if not visited[i]:
                    cluster = []
                    stack = [i]
                    visited[i] = True
                    while stack:
                        node = stack.pop()
                        cluster.append(node)
                        for neighbor in adjacency[node]:
                            if not visited[neighbor]:
                                visited[neighbor] = True
                                stack.append(neighbor)
                    clusters.append(cluster)

            average_boxes_list = []
            for cluster in clusters:
                cluster_boxes = [labels[idx] for idx in cluster]
                avg_box = average_boxes(cluster_boxes)
                average_boxes_list.append(avg_box)

            input_labels[img_path] = average_boxes_list

    if epoch > specified_epoch:
        IoU_threshold_cluster = 0.3 
        for img_path in candidate_labels:
            labels = candidate_labels[img_path]  
            if len(labels) < 3:
                continue

            num_labels = len(labels)
            adjacency = [[] for _ in range(num_labels)]

            for i in range(num_labels):
                for j in range(i + 1, num_labels):
                    box1 = labels[i][:4] 
                    box2 = labels[j][:4]
                    iou = compute_iou(box1, box2)
                    if iou >= IoU_threshold_cluster:
                        adjacency[i].append(j)
                        adjacency[j].append(i)

            visited = [False] * num_labels
            clusters = []

            for i in range(num_labels):
                if not visited[i]:
                    cluster = []
                    stack = [i]
                    visited[i] = True
                    while stack:
                        node = stack.pop()
                        cluster.append(node)
                        for neighbor in adjacency[node]:
                            if not visited[neighbor]:
                                visited[neighbor] = True
                                stack.append(neighbor)
                    clusters.append(cluster)

            for cluster in clusters:
                if len(cluster) >= 3:
                    cluster_boxes = [labels[idx] for idx in cluster]  
                    representative_box = cluster_boxes[0][:4]
                    input_boxes = []
                    if img_path in input_labels:
                        input_boxes = input_labels[img_path] 

                    has_match = False
                    for input_box in input_boxes:
                        iou_with_input = compute_iou(representative_box, input_box)
                        if iou_with_input > iou_threshold:
                            has_match = True
                            break
                    if not has_match:
                        if img_path not in input_labels:
                            input_labels[img_path] = []
                        input_labels[img_path].append(representative_box)
    else:
        pass

    try:
        with open(input_file, 'w') as f_output:
            for img_path in input_labels:
                labels = input_labels[img_path]
                if labels:
                    labels_str_list = []
                    for label in labels:
                        x1, y1, x2, y2 = label
                        labels_str_list.append(f"{x1},{y1},{x2},{y2},0")
                    labels_str = ' '.join(labels_str_list)
                    f_output.write(f"{img_path} {labels_str}\n")
                else:
                    f_output.write(f"{img_path}\n")
    except Exception as e:
        print(f"Error writing to {input_file}: {e}")
        return

    # Write back updated candidate_labels to candidate_file
    try:
        with open(candidate_file, 'w') as f_candidate:
            for img_path in candidate_labels:
                labels = candidate_labels[img_path]
                if labels:
                    labels_str_list = []
                    for label in labels:
                        x1, y1, x2, y2, count = label
                        labels_str_list.append(f"{x1},{y1},{x2},{y2},{count}")
                    labels_str = ' '.join(labels_str_list)
                    f_candidate.write(f"{img_path} {labels_str}\n")
                else:
                    f_candidate.write(f"{img_path}\n")
    except Exception as e:
        print(f"Error writing to {candidate_file}: {e}")
        return


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




def augment_pseudo_labels_with_motion(input_file, output_dir, augmented_input_file, num_targets_per_sequence):
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
                final_bbox = (overlay_x1_clamped, overlay_y1_clamped, overlay_x2_clamped+1, overlay_y2_clamped+1) 
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


        
def filter_targets(input_file: str, dist_thresh=30, min_track_len=6):
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
