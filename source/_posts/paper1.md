---
title: paper1
date: 2020-07-12 13:30:02
tags: Paper reading notes in Week1
mathjax: true
categories: paper notes
---

# 写在前面

本篇论文《Dual Super-Resolution Learning for Semantic Segmentation》是中科院和其他公司合作的一篇结合超分辨和语义分割的文章,文章发表在2020CVPR.

Github地址: https: //github.com/wanglixilinx/DSRL(未开源)

* 数据集：CityScapes、COCO、MS
* 解决问题：对于同一输入,分辨率越高,语义分割的精度也越高,但导致计算量的增加,作者的出发点在于从低分辨的输入得到较为精准的高分辨率语义分割。
* 核心思想：将超分辨任务与语义分割相结合。超分辨任务在重构图像时,会学到较好的高层次的语义特征,那么怎样从低分辨率恢复至高分辨率,作者提出用超分辨任务去指导语义分割分辨率的恢复(适用于计算资源受限的移动设备)。

<!--more-->

# Introduction

​	开篇作者就给了实验结果(如下图),证明对于任何的语义分割网络,高分辨率的输入都要比低分辨率输入的分割结果好。

{% asset_img 1.png %}

​	在图中我们可以直观地看到当图像分辨率从256×512变为512×1024时,mIoU值平均增加了10%以上。也就是说一旦限制了输入的 大小,无论大型网络还是轻量级网络其性能都会下降,对于语义分割这类逐像素的分类任务而言,如何做到同时保持高效的推理速度和出色的性能表现是一个挑战。

​	现有的保持高分辨率输出的两类方法,第一种像U-Net这种Encoder-Decoder结构中间加skip-connection,第二类像Deeplabv3+利用空洞卷积代替标准卷积,但它们开销大,一方面输入必须是高分辨率图像,另一方面计算资源消耗大(具体表现在FLOPs上,这里补充一个小的知识点:`FLOPS指的是每秒浮点数运算次数,可以理解为是计算速度,是衡量硬件性能的指标,而FLOPs指浮点数运算量用来衡量模型/算法的复杂度`)。但是呢轻量级网络虽然计算量小,但是在性能表现上面远低于SOTA的效果。

# Contributions

​	针对上面的问题,作者在本篇中有了如下的方法贡献:

1. 提出双重超分辨率学习(DSRL)来保持高分辨率表示。DSRL包含了三个模块分别是SSSR、SISR、FA。具体来说,将超分辨的思想整合到现有的语义分割的pipeline中。
2. 验证了DSRL框架的泛化性,可以很容易扩展到其他任务中。
3. 实验证明了该方法在语义分割和人体姿势估计方面的有效性。使用相似的计算量,性能可以提高>=2%;达到相当的性能需要的FLOPs减少30%。

# Methods

* 网络组成

  {% asset_img 2.png %}

  * SSSR

    ​	我们来分析一下这个网络结构：上面一条分支SSSR是mainline,下面一部分是SISR,用于将低分辨率图像reconstruct成超分辨率图像(MSE Loss指导网络训练),中间一部分是FA模块。

    ​	对于语义分割,只需要附加一个额外的upsampling模块(由一堆反卷积组成)就可以产生最终的预测mask,整个过程称为语义分割超分辨率,然后是BatchNorm和ReLU层,只需要较少的参数。

  * SISR

    仅靠解码器模块还不足以恢复使用原始图像作为输入获得的类似高分辨率的语义特征。由于解码器是简单的双线性上采样或者子网络,由于输入图片的分辨率较低,因此不会包含其他任何信息。

    ​	SISR旨在从低分辨率输入中重建高分辨率图像。这意味着SISR可以在低分辨率输入下有效地重建图像的细粒度结构信息,这对于语义分割总是有帮助的(相当于提供了更多的特征值了嘛)。为了显示更好地理解,下图中可视化了SSSR和SISR的功能,同过对比可以发现SISR包含更完整的对象结构。尽管这些结构没有明确的类别,但是可以通过像素与像素之间的关系有效地对它们进行分组。众所周知这些关系可以隐式地传递语义信息,从而有利于语义分割的任务。

    {% asset_img 3.png %}

    对于SISR分支,它与SSSR共享特征提取器。见下图：

    {% asset_img 4.png %}

  * FA

    ​	因为SISR比SSSR包含更多完整结构的信息,用此模块来指导SSSR学习高分辨率的表征。FA旨在学习SISR和SSSR分支之间的相似度矩阵的距离,其中相似度矩阵主要描述像素之间的成对关系。

    ​	相似度矩阵的定义：

​                                                              $$S_{ij} = (\frac{F_i}{\Vert F_i \Vert_{p}})^T·\frac{F_j}{\Vert F_j \Vert_{p}})$$

​					衡量SSSR和SISR之间相似矩阵的距离：$\lambda$

​                                                   $$ L_{fa} = \frac{1}{W^{'2}H^{'2}} \mathop{\Sigma}\limits_{i = 1}^{W^{'}H^{'}} \mathop{\Sigma}\limits_{j = 1}^{W^{'}H^{'}} \Vert S_{ij}^{seg} - S_{ij}^{sr} \Vert_{q}$$

# LossFunction

* total loss function

    $$ L = L_{ce} + \omega_{1}L_{mse} + \omega_{2}L_{fa}$$

* 用于SSSR的交叉熵损失$$L_{ce}$$

  $$L_{ce} = \frac{1}{N} \mathop{\Sigma}\limits_{i = 1}^{N}-y_ilog{p_i}$$

* 用于SISR的均方误差损失$$L_{mse}$$

  $$L_{mse} = \frac{1}{N} \mathop{\Sigma}\limits_{i = 1}^{N} \Vert SISR(X_{i}) - Y_{i}\Vert^2$$

# Experiment

* 消融实验

  ​	作者分别在轻量级网络和大型网络ESPNet和Deeplabv3+上进行了实验,并在Cityscapes得到了验证,评价指标是mIoU.

  {% asset_img 5.png %}

* Generity

  作者也用实验证明了开篇Contributions中提到的泛化性,网络同样可以用到行人姿态检测中,并且可以取得不错的实验效果,如下图所示。

  ​	{% asset_img 6.png %}

# Inspiration

* 发现有效的问题很重要

  看到这篇paper的原因就是现在在做的项目存在图片分辨率低的问题,正是实际应用中存在、亟待解决的问题,有时候能提出好问题比有好的解决方法更重要。这篇就是解决在资源受限的移动设备上进行seg的任务(要不然能有高分辨率图像当然用富含更多信息的高分辨率图像呀！)

* 多任务相结合

  上一篇看到的就是类似将边缘检测和语义分割相结合的一种方式(《ET-Net: A Generic Edge-aTtention Guidance Network for Medical Image Segmentation》)这篇说白了就是将超分辨和语义分割结合的一个网络,所以我觉得平时也要多读一下其他方向的paper,说不定什么时候就能用到了呢！