---
title: paper2
top: false
cover: false
toc: true
mathjax: true
date: 2020-07-16 09:56:58
password:
summary:
tags:
categories:
---

# 写在前面

本篇论文《Deep Cascaded Attention Network for Multi-task Brain Tumor Segmentation》是中科大和其他高校合作的一篇结合attention机制的文章,文章发表在2020MICCAI.

* Github地址: 未给出

* 数据集：BraTS challenge 2018
* 解决问题：用cascaded model解决muti-task的问题会造成model redundancy/train complexity/task isolation
* 本文贡献：
  * 1.将subtask分解为不同的branches(实质上仍为single model),也就是说再一些特征提取时可以共享extractor,从而减少参数降低模型的复杂性;
  * 2.BraTS challenge是一个分级的分割任务(https://zhuanlan.zhihu.com/p/159202689 对数据集和此任务的解释可以看我知乎文章),所以作者想到把Cascaded Attention Module(以下简称CAM)加到branch之间;
  * 3.作者认为backbone——3D-Unet的skip-connection非常低效,又因为原来concat的两个feature存在较大的gap,所以作者用FBM替代原来的skip-connection,既能缓解feature fusion gap(也可以换一种说法是指导特征融合),也能获取层间关系;
  * 4.作者可视化了CAM以证明该模块的有效性。

<!--more-->

# 1 Introduction

​	BraTS challenge是对大脑不同时期的胶质瘤进行分割。MRI有多种模态,不同模态的侧重点是不一样的,比如说T1序列可能侧重于结构信息,T2序列比较侧重于病灶信息(我瞎猜的),因为异构的多样性,所以放射科医生对胶质瘤的评估需要专业的经验且这种经验因医生而异,因此有一套自动化的分割模型是对病人情况的诊断是非常必要的。

​	作者认为对于脑肿瘤分割任务,well-designed的模型设计很重要,但一些先验知识(比如肿瘤子区域的层间关系)一样有助于多任务的分割。作者比较了利用先验知识进行分割的两种网络“Model Cascaded”(MC) strategy和One-pass Multi-task Network(OM-Net),它们采取的做法是把三个任务整合到一个网络,再通过curriculum learning(课程学习,先学习简单的再学习困难的)的方法。

​	这样做的弊端在于：1. 增加模型的复杂度:因为三个任务的模型几乎是完全一样的;2.将multi-task通one-by-one的方式依次训练就会导致后面模型训练的质量非常依赖于前面模型训练的效果。同时值得我们注意的是,CNN的feature包含了非常多的位置(localization info我不太知道怎么翻译)信息可以被充分利用。

​	此外作者在BraTS challenge2018数据集上取得了很好的指标性结果,只用了1/4最好模型的参数量却达到了和它相当的分割效果。baseline选用的是NoNewNet(即nnNet),NoNewNet+是用了额外的数据集的。

# 2 Method

### Backbone Network Architecture

我们先来看此网络的backbone：

{% asset_img 1.png %}

这可以视为3D Res-Unet,3D网络网络更多地利用了连续切片的一致性，本质上包含了三维卷积核的多视图融合。为了避免梯度发散加速模型收敛,残差连接被应用其中.残差块采用bottleneck的结构(即先减少通道数,再卷积最后再还原通道数)。

### Feature Bridge Module

可以从上图看出,此网络和最原始的3D Unet最大的不同点在于skip-connection被更改了。作者认为直接把low-level和high-level即spatial feature&semantic feature结合是lack efficiency的,FBM级联了2.5D Residual block和3D block去平衡切片之间和切片内部的权重。

### Deep Cascaded Attention Network

因为本次分割任务是分层次的,三个分割任务,但有包含关系见下图:

{% asset_img 0.jpg %}

前面提到的MC和OM-Net通过one by one的训练策略充分利用了这层次关系,但这样的做法会导致model redundancy和task isolation。为解决这个问题作者提出了DCAN,网络结构如下图所示：

{% asset_img 2.png %}

作者认为对于多任务,低层次的feature map主要包含灰度或者边缘信息,不同的任务在low level这里只有很小的差异但在high-level这里差异就很明显了,所以在low-level这里三个branches共用了一个feature extractor。每个branch都以前面提到的Res-Unet(加FBM)为backbone,对每个分支都把最后的max-pooling换成了dilated conv,这样做的原因是增大感受野减少信息丢失

### Cascaded Attention Module

high-level包含丰富的位置信息,为了充分利用这些特征信息,引入CAM去提取位置信息隐式地指导后面branch训练

(这样的话其实跟cascade差不了太多,just最开始的特征提取器是三个branch共享的)

从上面的网络结构图可以看出：

{% asset_img 3.png %}

F()表示上图右边的Residual block+Sigmoid()生成的attention map(最开始的时候作者在branch之间加入Attention也就是$X_{et}$来自$X_s$和$Y_{tc}$ 但发现效果不是特别好$Y_{tc}$不能提供足够的location info,所以后来作者先concat($Y_{wt}Y_{tc}$)再得到Attention map)

### Visualization of Cascaded Attention map

{% asset_img 4.png %}

Attention可视化结果由上面公式实现,可视化结果如下图:

{% asset_img heatmap.png %}

# 3 Experiments

数据集是BraTS challenge 2018,采用mirror/random scaling/random rotate/random shift进行数据增强,损失函数采用Diceloss+Focalloss

{% asset_img lossfunction.png %}

指标性结果：

{% asset_img exp1.png %}

{% asset_img exp2.png %}

# 4 Conclusion

* 利用分割子区域间的包含关系作为先验信息;
* DCAN利用层间级联attention机制同时完成多任务而不是将multi-task问题拆成多个连续的model;
* AttentionMap的可视化结果表明了CAM的有效性,上个branch的结果确实可以帮助后面branch进行训练;
* FBM缓解了long-range feature fusion gap;
* 用1/4参数达到了和最优模型相当的结果。