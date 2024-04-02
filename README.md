# MSMTSeg: Multi-Stained Multi-Tissue Segmentation of Kidney Histology Images via Generative Self-Supervised Meta-Learning Framework
<br>**[Xueyu Liu](https://scholar.google.com.hk/citations?user=jeatLqIAAAAJ&hl=zh-CN), Rui Wang, Yexin Lai, Yongfei Wu, Hangbei Cheng, Yuanyue Lu, Jianan Zhang, Ning Hao, Chenglong Ban, Yanru Wang, Shuqin Tang, Yuxuan Yang, Ming Li, Xiaoshuang Zhou and Wen Zheng**<br>



We present a generative self-supervised meta-learning framework to implement multi-stained multi-tissue segmentation, namely MSMTSeg, from renal biopsy WSIs using only a few annotated samples for each stain domain. MSMTSeg consists of multiple stain transform models for achieving inter-translation between multiple stain domains, a self-supervision module to obtain a pre-trained weight with individual information for each stain, and a meta-learning strategy combined with generated virtual data and the pre-trained weights to learn the common information across multiple stain domains and improve segmentation performance. 

<p align="center">
<img width="800" alt="ablation" src="img/workflow.png">
</p>

## To do
The complete code will be made public later

## Data available
We have initiated a data access review process of our work. Interested parties can now request permission to access the data by emailing the address: wuyongfei@tyut.edu.cn.

## Acknowledgement
Thanks [Cycle-Gan](https://github.com/junyanz/CycleGAN), [MoCov3](https://github.com/CupidJay/MoCov3-pytorch), [MAML](https://github.com/dragen1860/MAML-Pytorch), [Unet](https://github.com/milesial/Pytorch-UNet). for serving as building blocks of MSMTSeg.
