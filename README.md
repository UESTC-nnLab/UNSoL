## ***The first end-to-end unsupervised framework for moving infrared small target detection***

This project is the implementation of our paper **Unsupervised Self-optimization Learning for Moving Infrared Small Target Detection**, currently under review.



## Datasets (bounding box-based)
- Datasets are available at [`DSISTD`](https://pan.baidu.com/s/1-di7v8e1Vmp3PzzRqEGKHg?pwd=r5cg)(code: r5cg) and [`IRDST-M`](https://pan.baidu.com/s/1jGW76xbt30XuA9A-YfUz-Q?pwd=w6pc)(code: w6pc). 


- You need to reorganize these datasets in a format similar to the `train.txt` and `val.txt` files we provided (`.txt files` are used in training).  We provide the `.txt files` for DSISTD and IRDST-M.
For example:
```python
train_annotation_path = '/home/DSISTD/train.txt'
val_annotation_path = '/home/DSISTD/val.txt'
```
- Or you can generate a new `txt file` based on the path of your datasets. `.txt files` (e.g., `train.txt`) can be generated from `.json files` (e.g., `instances_train2017.json`). We also provide all `.json files` for [`DSISTD`](https://pan.baidu.com/s/1-di7v8e1Vmp3PzzRqEGKHg?pwd=r5cg)(code: r5cg) and and [`IRDST-M`](https://pan.baidu.com/s/1jGW76xbt30XuA9A-YfUz-Q?pwd=w6pc)(code: w6pc). 

``` python 
python utils_coco/coco_to_txt.py
```

- The folder structure should look like this:
```
DSISTD
├─train.json
├─test.json
├─train.txt
├─val.txt
├─test.txt
├─1
│   ├─0.bmp
│   ├─1.bmp
│   ├─2.bmp
│   ├─ ...
├─2
│   ├─0.bmp
│   ├─1.bmp
│   ├─2.bmp
│   ├─ ...
├─3
│   ├─ ...
```


## Prerequisite

* python==3.11.8
* pytorch==2.1.1
* torchvision==0.16.1
* numpy==1.26.4
* opencv-python==4.9.0.80
* scipy==1.13
* Tested on Ubuntu 20.04, with CUDA 11.8, and 1x NVIDIA 3090.


## Usage of UNSoL

### Path Configuration (using DSISTD as an example)



- #### label_generation_DSISTD.py

Modify the following paths according to your dataset location:
```python
# Path to your datasets

path_file = '/home/public/DSISTD/train.txt'
final_file = '/home/public/DSISTD/train_pseudo_labels.txt'
candidate_file = '/home/public/DSISTD/candidate_pseudo_labels.txt'
aug_file = '/home/public/DSISTD/train_pseudo_labels_0.txt'
aug_dir = '/home/public/DSISTD_AUG'
```
- #### train_phase0_DSISTD.py
The data required for the first training stage uses the previously generated pseudo-labels, modify the corresponding path accordingly:
```python
train_annotation_path = '/home/public/DSISTD/train_pseudo_labels_0.txt'
val_annotation_path = '/home/public/DSISTD/val.txt'
```

- #### train_phase1_DSISTD.py
The data used in the second training stage consists of pseudo-labels iteratively refined through self-optimization learning.
```python
train_annotation_path = '/home/public/DSISTD/train_pseudo_labels.txt'
val_annotation_path = '/home/public/DSISTD/val.txt'
```

- #### train_sup_DSISTD.py
The data used in the final training stage consists of the fully optimized pseudo-labels.
```python
train_annotation_path = '/home/public/DSISTD/train_pseudo_labels.txt'
val_annotation_path = '/home/public/DSISTD/val.txt'
```

- #### utils_fit_phase0_DSISTD.py and utils_fit_phase1_DSISTD.py
Modify the paths required for the pseudo-label optimization process.
```python
input_file = '/home/public/DSISTD/train_pseudo_labels.txt'
candidate_file = '/home/public/DSISTD/candidate_pseudo_labels.txt'
path_file = '/home/public/DSISTD/train.txt'
output_dir = '/home/public/DSISTD_AUG'
```

### Training
- Note: Please use different `dataloader` for different datasets. For example, to train the model on DSISTD dataset, enter the following command: 
```python
CUDA_VISIBLE_DEVICES=0 python unsup_train_DSISTD.py 
```

### Test
- Usually `model_best.pth` is not necessarily the best model. The best model may have a lower val_loss or a higher AP50 during verification.
```python
"model_path": '/home/MoPKL/logs/model.pth'
```
- You need to change the path of the `json file` of test sets. For example:
```python
# Use DSISTD dataset for test

cocoGt_path         = '/home/public/DSISTD/test.json'
dataset_img_path    = '/home/public/DSISTD/'
```
```python
python test.py
```

### Visulization
- We support `video` and `single-frame image` prediction.
```python
# mode = "video" (predict a sequence)

mode = "predict"  # Predict a single-frame image 
```
```python
python predict.py
```



## Results
- For bounding box detection, we use COCO's evaluation metrics:

<!-- Results: DSISTD (left) vs IRDST-M (right) -->
<style>
  .up   { color: #2ca02c; font-weight: 600; }   /* green */
  .down { color: #d62728; font-weight: 600; }   /* red   */
  table.results { border-collapse: collapse; width: 100%; }
  table.results th, table.results td { border: 1px solid #ddd; padding: 6px 8px; }
  table.results th { background:#f6f6f6; }
  .subhead { background:#fbfbfb; font-weight: 600; text-align:center; }
  .center { text-align:center; }
</style>

<h3>Results</h3>
<p>For bounding box detection, we use COCO metrics.</p>

<table class="results">
  <tr>
    <th rowspan="2">Method</th>
    <th rowspan="2">Backbone / Setting</th>
    <th colspan="4">DSISTD</th>
    <th colspan="4">IRDST-M</th>
    <th rowspan="2">Download</th>
  </tr>
  <tr>
    <th>mAP50 (%)</th><th>Precision (%)</th><th>Recall (%)</th><th>F1 (%)</th>
    <th>mAP50 (%)</th><th>Precision (%)</th><th>Recall (%)</th><th>F1 (%)</th>
  </tr>

  <!-- ================= UNSoL on ACM ================= -->
  <tr class="subhead"><td colspan="10">UNSoL on ACM</td></tr>
  <tr>
    <td class="center">UNSoL</td>
    <td class="center">ACM</td>
    <td class="center">57.16</td><td class="center">80.65</td><td class="center">71.18</td><td class="center">75.62</td>
    <td class="center">69.08</td><td class="center">77.32</td><td class="center">90.27</td><td class="center">83.30</td>
    <td rowspan="7" class="center">
      <a href="https://pan.baidu.com/s/18O7gEwr-QvMxrckCJrRH_Q?pwd=2u4k">Baidu</a> (code: 2u4k)
    </td>
  </tr>
  <tr>
    <td class="center">Gain (Δ)</td>
    <td class="center">vs. baseline</td>
    <td class="center"><span class="down">↓9.69</span></td>
    <td class="center"><span class="down">↓9.69</span></td>
    <td class="center"><span class="down">↓5.45</span></td>
    <td class="center"><span class="down">↓6.45</span></td>
    <td class="center"><span class="down">↓3.92</span></td>
    <td class="center"><span class="down">↓11.18</span></td>
    <td class="center"><span class="up">↑6.93</span></td>
    <td class="center"><span class="down">↓2.54</span></td>
  </tr>

  <!-- ================= UNSoL on MSHNet ================= -->
  <tr class="subhead"><td colspan="10">UNSoL on MSHNet</td></tr>
  <tr>
    <td class="center">UNSoL</td>
    <td class="center">MSHNet</td>
    <td class="center">71.22</td><td class="center">91.60</td><td class="center">78.51</td><td class="center">84.55</td>
    <td class="center">72.92</td><td class="center">80.71</td><td class="center"><b>91.15</b></td><td class="center">85.57</td>
  </tr>
  <tr>
    <td class="center">Gain (Δ)</td>
    <td class="center">vs. baseline</td>
    <td class="center"><span class="up">↑0.50</span></td>
    <td class="center"><span class="up">↑8.15</span></td>
    <td class="center"><span class="down">↓6.84</span></td>
    <td class="center"><span class="up">↑0.16</span></td>
    <td class="center"><span class="up">↑0.40</span></td>
    <td class="center"><span class="down">↓7.16</span></td>
    <td class="center"><span class="up">↑7.51</span></td>
    <td class="center"><span class="down">↓0.13</span></td>
  </tr>

  <!-- ================= UNSoL on RDIAN ================= -->
  <tr class="subhead"><td colspan="10">UNSoL on RDIAN</td></tr>
  <tr>
    <td class="center">UNSoL</td>
    <td class="center">RDIAN</td>
    <td class="center"><b>73.84</b></td><td class="center"><b>93.86</b></td><td class="center">79.33</td><td class="center">85.99</td>
    <td class="center">70.31</td><td class="center">86.50</td><td class="center">82.75</td><td class="center">84.58</td>
  </tr>
  <tr>
    <td class="center">Gain (Δ)</td>
    <td class="center">vs. baseline</td>
    <td class="center"><span class="up">↑1.86</span></td>
    <td class="center"><span class="up">↑6.61</span></td>
    <td class="center"><span class="down">↓4.23</span></td>
    <td class="center"><span class="up">↑0.63</span></td>
    <td class="center"><span class="down">↓6.04</span></td>
    <td class="center"><span class="down">↓2.89</span></td>
    <td class="center"><span class="down">↓3.90</span></td>
    <td class="center"><span class="down">↓3.42</span></td>
  </tr>
</table>




- Pseudo-label evolution and the unsupervised self-optimization learning process on DSISTD.

<img src="/learning.pdf" width="700px">


## Contact
If any questions, kindly contact with Shengjia Chen via e-mail: csj_uestc@126.com.

## References
1. S. Chen, L. Ji, J. Zhu, M. Ye and X. Yao, "SSTNet: Sliced Spatio-Temporal Network With Cross-Slice ConvLSTM for Moving Infrared Dim-Small Target Detection," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-12, 2024, Art no. 5000912, doi: 10.1109/TGRS.2024.3350024. 
2. Bingwei Hui, Zhiyong Song, Hongqi Fan, et al. A dataset for infrared image dim-small aircraft target detection and tracking under ground / air background[DS/OL]. V1. Science Data Bank, 2019[2024-12-10]. https://cstr.cn/31253.11.sciencedb.902. CSTR:31253.11.sciencedb.902.
3. Ruigang Fu, Hongqi Fan, Yongfeng Zhu, et al. A dataset for infrared time-sensitive target detection and tracking for air-ground application[DS/OL]. V2. Science Data Bank, 2022[2024-12-10]. https://cstr.cn/31253.11.sciencedb.j00001.00331. CSTR:31253.11.sciencedb.j00001.00331.

