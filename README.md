# 🔥🔥 Pick of the Bunch: Detecting Infrared Small Targets Beyond Hit-Miss Trade-Offs via Selective Rank-Aware Attention 🔥🔥 

Yimian Dai, Peiwen Pan, Yulei Qian, Yuxuan Li, Xiang Li, Jian Yang, Huan Wang

This repository is the official site for "Pick of the Bunch: Detecting Infrared Small Targets Beyond Hit-Miss Trade-Offs via Selective Rank-Aware Attention".

## Abstract

Infrared small target detection faces the inherent challenge of precisely localizing dim targets amidst complex background clutter. Traditional approaches struggle to balance detection precision and false alarm rates. To break this dilemma, we propose SeRankDet, a deep network that achieves high accuracy beyond the conventional hit-miss trade-off, by following the ``Pick of the Bunch'' principle. At its core lies our Selective Rank-Aware Attention (SeRank) module, employing a non-linear Top-K selection process that preserves the most salient responses, preventing target signal dilution while maintaining constant complexity. Furthermore, we replace the static concatenation typical in U-Net structures with our Large Selective Feature Fusion (LSFF) module, a dynamic fusion strategy that empowers SeRankDet with adaptive feature integration, enhancing its ability to discriminate true targets from false alarms. The network's discernment is further refined by our Dilated Difference Convolution (DDC) module, which merges differential convolution aimed at amplifying subtle target characteristics with dilated convolution to expand the receptive field, thereby substantially improving target-background separation. Despite its lightweight architecture, the proposed SeRankDet sets new benchmarks in state-of-the-art performance across multiple public datasets. The code is available at <https://github.com/GrokCV/SeRankDet>.

- [Abstract](#abstract)
- [Installation](#installation)
  - [Step 1: Create a conda environment](#step-1-create-a-conda-environment)
  - [Step 2: Install PyTorch](#step-2-install-pytorch)
  - [Step 3: Install OpenMMLab Codebases](#step-3-install-openmmlab-codebases)
- [Dataset Preparation](#dataset-preparation)
  - [File Structure](#file-structure)
  - [Datasets Link](#datasets-link)
- [Training](#training)
  - [Single GPU Training](#single-gpu-training)
  - [Multi GPU Training](#multi-gpu-training)
  - [Notes](#notes)
- [Test](#test)
- [Model Zoo and Benchmark](#model-zoo-and-benchmark)
  - [Leaderboard](#leaderboard)
  - [Model Zoo](#model-zoo)
- [Citation](#citation)



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

The datasets used in this project and the dataset split files can be downloaded from the following links:

* NoisySIRST Dataset
  * [Baidu Netdisk](https://pan.baidu.com/s/15RUYw23RSC20Xk1c1dMKYA?pwd=grok)
  * [OneDrive](https://1drv.ms/f/s!AmElF7K4aY9pgYEae4JdbbMd--tzNQ?e=yKwxa3)
* SIRST Dataset
  * [Baidu Netdisk](https://pan.baidu.com/s/1LgnBKcE8Cqlay5GnXfUaLA?pwd=grok)
  * [OneDrive](https://1drv.ms/f/s!AmElF7K4aY9pgYEgG0VEoH3nDbiWDA?e=gkUW2W)
* SIRST-AUG Dataset
  * [Baidu Netdisk](https://pan.baidu.com/s/1_kAocokYSclQNf_ZLWPIhQ?pwd=grok)
  * [OneDrive](https://1drv.ms/f/s!AmElF7K4aY9pgYEfdtbrZhLsbd0ITg?e=thyA6h)
* NUDT-SIRST Dataset
  * [Baidu Netdisk](https://pan.baidu.com/s/16BbL9H38cIcvaBh4tPNTCw?pwd=grok)
  * [OneDrive](https://1drv.ms/f/s!AmElF7K4aY9pgYEdBMrQDFM1Vi24DQ?e=vBNoN4)
* IRSTD1K Dataset
  * [Baidu Netdisk](https://pan.baidu.com/s/1nRoZu1eI9BLnpmsxw0Kdwg?pwd=grok)
  * [OneDrive](https://1drv.ms/f/s!AmElF7K4aY9pgYEepi2ipymni0amNQ?e=XZILFh)



<!-- The NoisySIRST dataset can be accessible via .
https://drive.google.com/drive/folders/1RGpVHccGb8B4_spX_RZPEMW9pyeXOIaC?usp=sharing -->

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
<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="2">SIRST</th>
    <th colspan="2">IRSTD1k</th>
    <th colspan="2">SIRSTAUG</th>
    <th colspan="2">NUDT-SIRST</th>
  </tr>
  <tr>
    <th>IoU</th>
    <th>nIoU</th>
    <th>IoU</th>
    <th>nIoU</th>
    <th>IoU</th>
    <th>nIoU</th>
    <th>IoU</th>
    <th>nIoU</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ACM</td>
    <td>72.88</td>
    <td>72.17</td>
    <td>63.39</td>
    <td>60.81</td>
    <td>73.84</td>
    <td>69.83</td>
    <td>68.48</td>
    <td>69.26</td>
  </tr>
  <tr>
    <td>RDIAN</td>
    <td>72.85</td>
    <td>73.96</td>
    <td>64.37</td>
    <td>64.90</td>
    <td>74.19</td>
    <td>69.80</td>
    <td>81.06</td>
    <td>81.72</td>
  </tr>
  <tr>
    <td>AGPCNet</td>
    <td>77.13</td>
    <td>75.19</td>
    <td>68.81</td>
    <td>66.18</td>
    <td>74.71</td>
    <td>71.49</td>
    <td>88.71</td>
    <td>87.48</td>
  </tr>
  <tr>
    <td>DNANet</td>
    <td>75.55</td>
    <td>75.90</td>
    <td>68.87</td>
    <td>67.53</td>
    <td>74.88</td>
    <td>70.23</td>
    <td>92.67</td>
    <td>92.09</td>
  </tr>
  <tr>
    <td>MTUNet</td>
    <td>78.75</td>
    <td>76.82</td>
    <td>67.50</td>
    <td>66.15</td>
    <td>74.70</td>
    <td>70.66</td>
    <td>87.49</td>
    <td>87.70</td>
  </tr>
  <tr>
    <td>UIUNet</td>
    <td>80.08</td>
    <td>78.09</td>
    <td>69.13</td>
    <td>67.19</td>
    <td>74.24</td>
    <td>70.57</td>
    <td>90.77</td>
    <td>90.17</td>
  </tr>
  <tr>
    <td>ABC</td>
    <td>81.01</td>
    <td>79.00</td>
    <td>72.02</td>
    <td>68.81</td>
    <td>76.12</td>
    <td>71.83</td>
    <td>92.85</td>
    <td>92.45</td>
  </tr>
  <tr>
    <td><strong>SeRankDet</strong></td>
    <td><strong>81.27</strong></td>
    <td><strong>79.66</strong></td>
    <td><strong>73.66</strong></td>
    <td><strong>69.11</strong></td>
    <td><strong>76.49</strong></td>
    <td><strong>71.98</strong></td>
    <td><strong>94.28</strong></td>
    <td><strong>93.69</strong></td>
  </tr>
</tbody>
</table>

### Model Zoo
Checkpoint and Train log: [BaiduCloud](https://pan.baidu.com/s/1iyv6Q8N23ywy1g6jGm9SLQ?pwd=grok)

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```bibtex
@article{dai2024SeRankDet,
  title={Pick of the Bunch: Detecting Infrared Small Targets Beyond Hit-Miss Trade-Offs via Selective Rank-Aware Attention},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  author={Dai, Yimian and Pan, Peiwen and Qian, Yulei and Li, Yuxuan and Li, Xiang and Yang, Jian and Wang, Huan},
  year={2024},
  volume={62},
  number={},
  pages={1-15}
}
```