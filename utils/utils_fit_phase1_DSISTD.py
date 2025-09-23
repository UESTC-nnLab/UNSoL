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



def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss        = 0

    ##################################
    input_file = '/home/public/DSISTD/train_pseudo_labels.txt'
    candidate_file = '/home/public/DSISTD/candidate_pseudo_labels.txt'
    path_file = '/home/public/DSISTD/train.txt'
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
        mapping_dict = {}
        with open(input_file, 'r') as f_input:
            input_lines = f_input.readlines()
            for line in input_lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    aug_img_path = parts[0]  
                    
                    orig_img_path = aug_img_path.replace("DSISTD_AUG", "DSISTD")
                    mapping_dict[orig_img_path] = aug_img_path
                   
        detector.net.eval()
        
        collected_image_paths = []
        collected_predictions = []
        
        with open(path_file, 'r') as f_path:
            train_lines = f_path.readlines()
        
        total_images = len(train_lines)
        pbar_eval = tqdm(total=total_images, desc="Collecting", mininterval=0.3)
        with torch.no_grad():
            for line in train_lines:
                parts = line.strip().split()
                if len(parts) == 0:
                    pbar_eval.update(1)
                    continue
        
                img_path = parts[0] 
                    
                try:
                    img_list_paths = get_history_imgs(img_path)
                    images = [Image.open(item) for item in img_list_paths]
                except Exception as e:
                    print(f"Failed to open image {img_path}, error: {e}")
                    pbar_eval.update(1)
                    continue
                _, boxes = detector.detect_image(images, crop=False, count=False)
                all_boxes = boxes.copy()
                all_boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]
                collected_predictions.append(all_boxes)
                aug_path = mapping_dict[img_path]
                collected_image_paths.append(aug_path)
                pbar_eval.update(1)
        pbar_eval.close()

        collect_pseudo_labels(collected_image_paths, collected_predictions, candidate_file, input_file)
        print('-------Candidate labels are collected-------')
        update_pseudo_labels(input_file, candidate_file, epoch+1, specified_epoch=5, iou_threshold=0.1)
        print('-------Labels update completed-------')



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
                    print(f"Warning: The prediction box format is incorrect and will be skipped. Prediction: {prediction}")
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

    for img_path in candidate_labels:
        labels = candidate_labels[img_path]
        new_labels = []
        for label in labels:
            x1, y1, x2, y2, count = label
            count = int(count) + 1 
            if count < 6:
                new_labels.append([x1, y1, x2, y2, count])
        candidate_labels[img_path] = new_labels

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


    if epoch >= specified_epoch:
        IoU_threshold_cluster = 0.3  
        for img_path in candidate_labels:
            labels = candidate_labels[img_path]  
            if len(labels) < 3:
                continue

            num_labels = len(labels)
            adjacency = [[] for _ in range(num_labels)]

            for i in range(num_labels):
                for j in range(i + 1, num_labels):
                    box1 = labels[i][:4]  # [x1, y1, x2, y2]
                    box2 = labels[j][:4]
                    iou = compute_iou(box1, box2)
                    if iou >= IoU_threshold_cluster:
                        # Add edge between i and j
                        adjacency[i].append(j)
                        adjacency[j].append(i)

            # Find connected components (clusters)
            visited = [False] * num_labels
            clusters = []

            for i in range(num_labels):
                if not visited[i]:
                    # Start a new cluster
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
                        # Append '0' at the end if needed
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




class TrackedObject:
    def __init__(self, bbox, object_id, frame_number):
        self.bbox = bbox  
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
                    print(f"Warning: Unable to read image {img_path}")
                    frame_data[sequence_number].append((frame_number, []))
                    pbar.update(1)
                    continue

                img_enhanced = cv2.equalizeHist(img)

                if previous_frame is not None:
                    kp, des = orb.detectAndCompute(img_enhanced, None)
                    if des is None or previous_des is None:
                        print(f"Warning: No descriptors found in frames {frame_number} or previous frame")
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
                                print(f"Warning: Homography computation failed for frame {frame_number}")
                                frame_diff = cv2.absdiff(img_enhanced, previous_frame)
                        else:
                            print(f"Warning: Not enough matches are found - {len(matches)} (minimum required is 4)")
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







def post_process_pseudo_labels(input_file, frame_data):

    class Track:
        def __init__(self, track_id, initial_position):
            self.track_id = track_id
            self.positions = deque([initial_position], maxlen=250) 
            self.velocities = deque(maxlen=10) 
            self.last_known_position = initial_position
            self.last_known_velocity = None
            self.frames_since_last_seen = 0

        def update(self, new_position):
            self.frames_since_last_seen = 0
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

    with open(input_file, 'w') as f_out:
        for sequence, frames in frame_data.items():
            tracks = []
            track_id_counter = 0
            max_frames_to_keep = 6
            distance_threshold = 80
            iou_threshold = 0.001  

            merge_iou_threshold = 0.01 

            frames = list(frames)
            frames_reversed = frames[::-1]  

            frame_outputs = [] 

            for i, (frame_number, targets) in enumerate(frames_reversed):
                img_path = os.path.join(os.path.dirname(str(input_file)), str(sequence), f"{frame_number}.bmp")
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Image not found: {img_path}")
                    continue

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
                        if cost_matrix[r][c] < distance_threshold:
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

                    unassigned_detections = [detection_centers[i] for i in range(len(detection_centers)) if
                                             i not in assigned_detections]
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
                        x1, y1, x2, y2 = max(1, x1), max(1, y1), min(img.shape[1] - 1, x2), min(img.shape[0] - 1, y2)
                        if 0 <= x1 <= img.shape[1] and 0 <= y1 < img.shape[0] and 0 <= x2 <= img.shape[1] and 0 <= y2 <= img.shape[0]:
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
                    output_lines = [f"{int(x1)},{int(y1)},{int(x2)},{int(y2)},0" for _, (x1, y1, x2, y2) in filtered_detections]
                    frame_outputs.append(f"{img_path} {' '.join(output_lines)}\n")
                else:
                    frame_outputs.append(f"{img_path}\n")

            for frame_output in frame_outputs[::-1]:
                f_out.write(frame_output)

    




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

    # 覆盖写回 input_file
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

