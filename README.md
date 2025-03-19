# DTA: Dual Temporal-channel-wise Attention for Spiking Neural Networks(

[WACV2025]: https://openaccess.thecvf.com/content/WACV2025/papers/Kim_DTA_Dual_Temporal-Channel-Wise_Attention_for_Spiking_Neural_Networks_WACV_2025_paper.pdf

# )

Code for DTA training



## Requirements

The environment for running this code has been tested and verified with the following versions:

* python 3.10
* pytorch 1.13.1
* cuda 11.6



## Training

To train the model, run the following script:

* bash ./scripts/run_`target dataset`.sh

Replace `<target_dataset>` with the name of the desired dataset (e.g., `cifar10/100`, `dvs_cifar10`, or `imgnet`).



## Dataset Structure

The expected directory structure for the datasets is as follows:

```
DTA-SNN/dataset/
├──CIFAR/
│	├──train/
│	├──val/
├──DVS_CIFAR10/
│	├──events_np/
│	├──extract/
│	├──frames_number_10_split_by_number/
├──ImageNet/
│	├──train/
│	├──val/
├── ......
```



## Citation

If you use this code in your work, please cite our paper:

```
@InProceedings{Kim_2025_WACV,
    author    = {Kim, Minje and Kim, Minjun and Yang, Xu},
    title     = {DTA: Dual Temporal-Channel-Wise Attention for Spiking Neural Networks},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {9682-9692}
}
```