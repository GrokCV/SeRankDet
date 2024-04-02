# Pick of the Bunch: Detecting Infrared Small Targets Beyond Hit-Miss Trade-Offs via Selective Rank-Aware Attention

- [Installation](#installation)
  - [Step 1: Create a conda environment](#step-1-create-a-conda-environment)
  - [Step 2: Install PyTorch](#step-2-install-pytorch)
  - [Step 3: Install OpenMMLab Codebases](#step-3-install-openmmlab-2x-codebases)
- [Dataset Preparation](#dataset-preparation)
  - [File Structure](#file-structure)
  - [Datasets Link](#datasets-link)
- [Training](#training)
  - [Single GPU Training](#single-gpu-training)
  - [Multi GPU Training](#multi-gpu-training)
- [Test](#test)
- [Model Zoo and Benchmark](#model-zoo-and-benchmark)
  - [Leaderboard](#leaderboard)
  - [Model Zoo](#model-zoo)
    - [Method A](#method-a)
    - [Method B](#method-b)


## Installation

### Step 1: Create a conda environment

```shell
$ conda create --name SeRankDet python=3.8
$ source activate SeRankDet
```

### Step 2: Install PyTorch

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=10.0 -c pytorch -c nvidia
```

### Step 3: Install OpenMMLab Codebases

```shell
# openmmlab codebases
pip install -U openmim
mim install mmcv-full==1.4.0
mim install mmdet==2.25.0
mim install mmsegmentation==0.28.0
```

**Note**: make sure you have `cd` to the root directory of `SeRankDet`

```shell
$ git clone git@github.com:GrokCV/SeRankDet.git
$ cd SeRankDet
```

## Dataset Preparation
### File Structure
```angular2html
|- datasets
   |- NUAA
      |-trainval
        |-images
          |-Misc_1.png
          ......
        |-masks
          |-Misc_1.png
          ......
      |-test
        |-images
          |-Misc_50.png
          ......
        |-masks
          |-Misc_50.png
          ......
   |-IRSTD1k
   |-NUDT
   |-SIRSTAUG

```
Please make sure that the path of your data set is consistent with the `data_root` in `configs/_base_/datasets/dataset_name.py`
### Datasets Link
https://drive.google.com/drive/folders/1RGpVHccGb8B4_spX_RZPEMW9pyeXOIaC?usp=sharing

## Training
### Single GPU Training

```
python train.py <CONFIG_FILE>
```

For example:

```
python train.py configs/unetseries/unetseries_serankdet_512x512_500e_irstd1k.py
```

### Multi GPU Training

```nproc_per_node``` is the number of gpus you are using.

```
python -m torch.distributed.launch --nproc_per_node=[GPU_NUMS] train.py <CONFIG_FILE>
```

For example:

```
python -m torch.distributed.launch --nproc_per_node=4 train.py configs/unetseries/unetseries_serankdet_512x512_500e_irstd1k.py
```

### Notes
* Be sure to set args.local_rank to 0 if using Multi-GPU training.

## Test

```
python test.py <CONFIG_FILE> <SEG_CHECKPOINT_FILE>
```

For example:

```
python test.py configs/unetseries/unetseries_serankdet_512x512_500e_irstd1k.py work_dirs/unetseries_serankdet_512x512_500e_irstd1k/20221009_231431/best_mIoU.pth.tar
```

If you want to visualize the result, you only add ```--show``` at the end of the above command.

The default image save path is under <SEG_CHECKPOINT_FILE>. You can use `--work-dir` to specify the test log path, and the image save path is under this path by default. Of course, you can also use `--show-dir` to specify the image save path.

## Model Zoo and Benchmark

**Note: Both passwords for BaiduYun and OneDrive is `grok`**.

### Leaderboard
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-7btt{border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" rowspan="2">Model</th>
    <th class="tg-c3ow" colspan="2">SIRST</th>
    <th class="tg-c3ow" colspan="2">IRSTD1k</th>
    <th class="tg-baqh" colspan="2">SIRSTAUG</th>
    <th class="tg-baqh" colspan="2">NUDT-SIRST</th>
  </tr>
  <tr>
    <th class="tg-c3ow">IoU</th>
    <th class="tg-c3ow">nIoU</th>
    <th class="tg-c3ow">IoU</th>
    <th class="tg-c3ow">nIoU</th>
    <th class="tg-baqh">IoU</th>
    <th class="tg-baqh">nIoU</th>
    <th class="tg-baqh">IoU</th>
    <th class="tg-baqh">nIoU</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">ACM</td>
    <td class="tg-c3ow">72.88</td>
    <td class="tg-c3ow">72.17</td>
    <td class="tg-c3ow">63.39</td>
    <td class="tg-c3ow">60.81</td>
    <td class="tg-baqh">73.84</td>
    <td class="tg-baqh">69.83</td>
    <td class="tg-baqh">68.48</td>
    <td class="tg-baqh">69.26</td>
  </tr>
  <tr>
    <td class="tg-c3ow">RDIAN</td>
    <td class="tg-c3ow">72.85</td>
    <td class="tg-c3ow">73.96</td>
    <td class="tg-c3ow">64.37</td>
    <td class="tg-c3ow">64.90</td>
    <td class="tg-baqh">74.19</td>
    <td class="tg-baqh">69.80</td>
    <td class="tg-baqh">81.06</td>
    <td class="tg-baqh">81.72</td>
  </tr>
  <tr>
    <td class="tg-c3ow">AGPCNet</td>
    <td class="tg-c3ow">77.13</td>
    <td class="tg-c3ow">75.19</td>
    <td class="tg-c3ow">68.81</td>
    <td class="tg-c3ow">66.18</td>
    <td class="tg-baqh">74.71</td>
    <td class="tg-baqh">71.49</td>
    <td class="tg-baqh">88.71</td>
    <td class="tg-baqh">87.48</td>
  </tr>
  <tr>
    <td class="tg-c3ow">DNANet</td>
    <td class="tg-c3ow">75.55</td>
    <td class="tg-c3ow">75.90</td>
    <td class="tg-c3ow">68.87</td>
    <td class="tg-c3ow">67.53</td>
    <td class="tg-baqh">74.88</td>
    <td class="tg-baqh">70.23</td>
    <td class="tg-baqh">92.67</td>
    <td class="tg-baqh">92.09</td>
  </tr>
  <tr>
    <td class="tg-c3ow">MTUNet</td>
    <td class="tg-c3ow">78.75</td>
    <td class="tg-c3ow">76.82</td>
    <td class="tg-c3ow">67.50</td>
    <td class="tg-c3ow">66.15</td>
    <td class="tg-baqh">74.70</td>
    <td class="tg-baqh">70.66</td>
    <td class="tg-baqh">87.49</td>
    <td class="tg-baqh">87.70</td>
  </tr>
  <tr>
    <td class="tg-c3ow">UIUNet</td>
    <td class="tg-c3ow">80.08</td>
    <td class="tg-c3ow">78.09</td>
    <td class="tg-c3ow">69.13</td>
    <td class="tg-c3ow">67.19</td>
    <td class="tg-baqh">74.24</td>
    <td class="tg-baqh">70.57</td>
    <td class="tg-baqh">90.77</td>
    <td class="tg-baqh">90.17</td>
  </tr>
  <tr>
    <td class="tg-c3ow">ABC</td>
    <td class="tg-c3ow">81.01</td>
    <td class="tg-c3ow">79.00</td>
    <td class="tg-c3ow">72.02</td>
    <td class="tg-c3ow">68.81</td>
    <td class="tg-baqh">76.12</td>
    <td class="tg-baqh">71.83</td>
    <td class="tg-baqh">92.85</td>
    <td class="tg-baqh">92.45</td>
  </tr>
  <tr>
    <td class="tg-7btt">SeRankDet</td>
    <td class="tg-7btt">81.27</td>
    <td class="tg-7btt">79.66</td>
    <td class="tg-7btt">73.66</td>
    <td class="tg-7btt">69.11</td>
    <td class="tg-amwm">76.49</td>
    <td class="tg-amwm">71.98</td>
    <td class="tg-amwm">94.28</td>
    <td class="tg-amwm">93.69</td>
  </tr>
</tbody>
</table>

### Model Zoo

