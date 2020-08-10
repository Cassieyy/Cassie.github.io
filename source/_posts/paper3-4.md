---
title: paper3&4
top: false
cover: false
toc: true
mathjax: true
date: 2020-07-17 20:57:05
password:
summary: Squeeze-and-Excitation, Image representation, Attention, CNN
tags: Attention mechanism
categories: paper notes
---

# 写在前面

Attention机制最早出现在NLP领域,最近在各个领域用的都很多,今天来详细介绍一下轻量级的SENet(Squeeze and Excitation Networks)和今年CVPR的ECA(Efficient Channel Attention)对SENet进行了改进。

论文链接：

SENet：https://arxiv.org/abs/1709.01507

ECA：https://arxiv.org/abs/1910.03151

<!--more-->

# 一.Squeeze-and-Excitation Networks(SENet)

Github:https://github.com/hujie-frank/SENet

SENet是ImageNet 2017(ImageNet收官赛)的冠军模型,和ResNet的出现类似,都在很大程度上减小了之前模型的错误率,并且复杂度低,新增参数和计算量小。作者的motivation在于,目前已经有很多工作在空间维度上来提升网络的性能。那么很自然想到，网络是否可以从其他层面来考虑去提升性能，比如考虑特征通道之间的关系。

## 1.Introduction

CNN靠局部感受野可以同时获取并整合spatial-wise和channel-wise的信息。对于不同的cv任务,我们更希望得到和此任务相关的图像的显著性特点(比如对于边缘检测,我希望图像的纹理信息得到增强),目前已经有很多工作通过learning mechaism获取spatial correlation以增强卷积抽取特征的能力。比如Inception系列的做法就是将multi-scale的思想加到网络中。本文中作者提出channel attention的思想,通过全局信息(加了FC层)重新对channel-wise进行权重的调整,SE block如下图所示。

{% asset_img 1.png %}

这跟Resnet有点像,但做的比residual block还要多,Resnet只是增加了一个skip connection，而SENet在相邻两层之间加入了处理，使得channel之间的信息交互成为可能，进一步提高了网络的准确率。

## 2.Squeeze-and-Excitation blocks

给定一个输入 x，其特征通道数为 c_1，通过一系列卷积等一般变换后得到一个特征通道数为 c_2 的特征。与传统的 CNN 不一样的是，接下来我们通过三个操作来重标定前面得到的特征。

#### 2.1Squeeze:Global Info Embedding

首先是 Squeeze 操作，我们顺着空间维度来进行特征压缩，将每个二维的特征通道变成一个实数，这个实数某种程度上具有全局(H*W)的感受野，并且输出的维度和输入的特征通道数相匹配。它表征着在特征通道上响应的全局分布，而且使得靠近输入的层也可以获得全局的感受野，这一点在很多任务中都是非常有用的。

公式表示：

{% asset_img 2.png %}

其实就是一个Global Avg Pooling,如果用pytorch的话,可以通过nn.AdaptiveAvgPool2d(1)实现.这一步骤可以得到C* 1* 1维度的weight0

#### 2.2Excitation:Adaptive Recalibration

其次是 Excitation 操作，它是一个类似于循环神经网络中门的机制。通过参数 w 来为每个特征通道生成权重，其中参数 w 被学习用来显式地建模特征通道间的相关性。

公式表示:

{% asset_img 3.png %}

这里用到的是类似bottleneck的设计,先是通过一个全连接层降维到C//r(r是一个自己设定的ratio,文中作者设为了16),在经过一个ReLU获得非线性关系,然后通过第二个全连接层还原到C维,最后是通过一个Sigmoid()将输出限制在(0, 1)之间。这样做比直接用一个 Fully Connected 层的好处在于：

1）具有更多的非线性，可以更好地拟合通道间复杂的相关性；

2）极大地减少了参数量和计算量。然后通过一个 Sigmoid 的门获得 0~1 之间归一化的权重，最后通过一个 Scale 的操作来将归一化后的权重加权到每个通道的特征上

#### 2.3Recalibration

最后是一个 Reweight 的操作，我们将 Excitation 的输出的权重看做是经过特征选择后的每个特征通道的重要性，然后通过乘法逐通道加权到先前的特征上，完成在通道维度上的对原始特征的重标定。

公式表示：

{% asset_img 4.png %}	

F_scale就是channel-wise的mul。

## 3.Generality

除此之外，SE 模块还可以嵌入到含有 skip-connections 的模块中。下图分别是将 SE 嵌入到 Inception和ResNet 模块中的例子，在ResNet的操作过程基本和 SE-Inception 一样，只不过是在 Addition 前对分支上 Residual 的特征进行了特征重标定。如果对 Addition 后主支上的特征进行重标定，由于在主干上存在 0~1 的 scale 操作，在网络较深 BP 优化时就会在靠近输入层容易出现梯度消散的情况，导致模型难以优化。

{% asset_img 5.png %}

目前大多数的主流网络都是基于这两种类似的单元通过 repeat 方式叠加来构造的。由此可见，SE 模块可以嵌入到现在几乎所有的网络结构中。通过在原始网络结构的 building block 单元中嵌入 SE 模块，我们可以获得不同种类的 SENet。如 SE-BN-Inception、SE-ResNet、SE-ReNeXt、SE-Inception-ResNet-v2 等等，证明SE有很好的泛化性。

## 4.Experiment

以下是SEblock在ImageNet上的表现。

{% asset_img 6.png %}

首先我们来看一下网络的深度对 SE 的影响。上表分别展示了 ResNet-50、ResNet-101、ResNet-152 和嵌入 SE 模型的结果。第一栏 Original 是原作者实现的结果，为了进行公平的比较，我们在 ROCS 上重新进行了实验得到 Our re-implementation 的结果（ps. 我们重实现的精度往往比原 paper 中要高一些）。最后一栏 SE-module 是指嵌入了 SE 模块的结果，它的训练参数和第二栏 Our re-implementation 一致。括号中的红色数值是指相对于 Our re-implementation 的精度提升的幅值。

从上表可以看出，SE-ResNets 在各种深度上都远远超过了其对应的没有 SE 的结构版本的精度，这说明无论网络的深度如何，SE 模块都能够给网络带来性能上的增益。值得一提的是，SE-ResNet-50 可以达到和 ResNet-101 一样的精度；更甚，SE-ResNet-101 远远地超过了更深的 ResNet-152。

## 5. Conclusion

* 是否每个Conv之后都用SE效果会好？应该是没有必要每个都用的,别忘了SE中含有两个FC,这会带来大量的参数引入。(还有关于这些block的位置问题,师兄说block的位置比较随便的,可能加到前面会稍微好一些);

* SENet的核心思想在于通过网络根据loss去学习特征权重，使得有效的feature map权重大，无效或效果小的feature map权重小的方式训练模型达到更好的结果,是一个channel-attention的思想.

# 二.ECA-Net:Efficient Channel Attention for Deep CNN

SENet是发表在2017年的,ECA是今年CVPR的一篇,主要是对SE进行了一些改进,下面我们一起来看一下。

## 1.Introducion

作者的motivation是“Can one learn effective channel attention in a more efficient way?“首先先来跟上面讲过的SENet进行一些对比,ECA作者认为1.SE在获取channel attention的模型还是太复杂了2.两个FC之间不应该用reduction ratio(这样的做法会影响channel attention prediction的准确性并且是unnecessary和inefficient的，目前所有的Spatial attention和Channel attention都存在模型复杂、计算量大的缺点)。

## 2.Method

#### 2.1 Avoiding Dimensionality Reduction

首先来看作者的对比实验：

{% asset_img 2.2.png %}

SE(GAP-FC(reduction)-ReLU-FC-Sigmoid)

SE-Var1(0参数的SE  GAP-Sigmoid)

SE-Var2(GAP-点乘-sigmoid)

SE-Var3(GAP-FC-sigmoid)

为了验证它的效果，作者比较了原始SE块和它的三个变体(即SE-Var1, SE-Var2和SEVar3)，所有这些都不执行维数降低(avoid dimensionality reduction)。从上表可以看出，无参数的SE-Var1仍然优于原始网络(Vanilla)，说明channel attention具有提高深度CNNs性能的能力。同时SE- var2独立学习各通道的权值，在参数较少的情况下略优于SE-Var1,这说明channel与其权值需要直接对应，而avoid降维比考虑非线性channel相关性更重要。

#### 2.2 Local Cross-Channel Interaction

{% asset_img 2.1.png %}

作者首先通过GroupConv实现即表中的SE-GC,组卷积可以学习到local cross-channel的特征,这一点下图可以很容易地看出来。

{% asset_img 2.3.png %}

每个神经元只连接输入的C_in//Group个channel的特征(这样做参数量是标准卷积的1/group),但实验发现这样的操作并没有带来效果的提升,作者分析原因可能是SE-GC丢弃了Group间的联系,只有Group间的cross-channel interactions了。

为了解决这个问题,作者用了含k个参数的Conv1d。为了更高效,作者使所有的channel共享这k个参数(ECA是在SE-Var2的基础上改的)。

## 3.Conclusion&Contributions

* 仔细分析了SEblock,避免dimension reduction对于effective CA(channel attention)是非常重要的;
* 适当的cross-channel interaction对于efficientCA是非常重要的;
* 提出ECA模块在增加可以忽略不计的参数量的同时带来明显性能的提升;
* 在ImageNet和COCO数据集上进行实验,比SOTA模型的复杂度低但是获得了competitive的表现。