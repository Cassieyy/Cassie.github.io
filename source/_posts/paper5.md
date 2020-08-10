---
title: paper5
top: false
cover: false
toc: true
mathjax: true
date: 2020-07-25 19:39:18
password:
summary:
tags: paper notes
categories: paper notes
---

## 写在前面

本篇论文《Non-local U-Nets for Biomedical Image Segmentation》是沈定刚老师团队的一篇将self-attention和seg结合的一篇paper,文章发表在2020AAAI.

论文地址：[https://arxiv.org/abs/1812.04103](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1812.04103)

代码实现(Pytroch)：[https://github.com/Whu-wxy/Non-local-U-Nets-2D-block](https://link.zhihu.com/?target=https%3A//github.com/Whu-wxy/Non-local-U-Nets-2D-block)

代码实现(Tensorflow)：[https://github.com/divelab/Non-local-U-Nets](https://link.zhihu.com/?target=https%3A//github.com/divelab/Non-local-U-Nets)

数据集(非开源)：3D多模态婴儿脑部MR图像

<!--more-->

## 分析问题

目前的Unet结构存在两个问题

- 四次down-sample参数量太多、下采样会丢失很多信息以及encoder的conv、pooling都是local算子，这里的local是相对于global而言的,我一次卷积操作感受野kernel_size大小,是局部(local)的,像Full-Connection就是全局(global)的,所以我想要获取global的信息就需要很深的编码器(感受野随conv的增加而增加);
- 上采样的过程涉及恢复图像的空间信息，如果只是局部信息而不考虑全局信息就很难做到。

## 解决方法

针对以上两个问题,作者给出了解决方法(想到需要global info我们可能首先会想到加FC Layer,但它参数量实在太大了,这也是CNN比FCN的先进性所在,现在更想去看看作者是怎么实现的啦！)

- 作者提出了一种基于self-attention的全局聚合块(global aggregation block)，使用这个block无需深度编码器就能聚合全局信息;
- 此block进一步扩展到up-sample。

## 整体网络结构

{% asset_img 1.png %}

首先我们先来看**整体**的网络结构,还是基于经典的Unet改的。up-sample和down-sample都是三次,skip-connection采用的是element-wise相加操作而不是concat操作,这样作者解释有两个优点：

- 减少trainable参数(因为feature map减半了);
- 加操作就像residual block,所以拥有残差块的所有优点。

## Residual blocks

讲完了上面Unet的整体结构,下面分解来讲上面结尾提到的residual block,作者提出了四种残差块：

{% asset_img 2.png %}

(a)是naive的residual block,(b)是用在down-sample的block(将原来的double-Conv3D+maxpooling换成了b),(c)是bottom-block(就是三次下采样三次上采样之间的block),(d)是up-sample的block(将原来转置卷积+double-Conv3D换成了d)

## Global Aggregation Block

如何不用full-connection而获取到global info呢,我们重点来讲一下基于self-attention的这个GAB,作者提到GAB可以融合任意维度的feature map的global info。下面是GAB的结构图：

{% asset_img 3.png %}

我们设X是block的输入维数是DHWC,Y是输出维数是D_qH_qW_qC_o,根据self-attention的做法：

- 首先如上图左半部分所示,并行生成了三个matrix。Query matrix是任意操作生成的;Key Vector是1*1*1Conv3D生成的,不改变spatial_size,这里要保证卷积的out_ch和Query matrix一样是C_k;Value matrix也是1*1*1Conv3D生成的,只不过out_ch是C_v而已,论文中也提到C_k和C_v都是超参数,根据自己的需要选取。 第一步生成了三个matrix之后,经过unfold操作,顾名思义unfold就是把上面三个matrix按channel维度展开(类似flatten操作)得到Q K V：

  {% asset_img 0.0.png %}

QueryTransformC_k(),Conv_1C_k(),Conv_1C_v()操作跟我上面提到的一样 只是被符号化表示了(我觉得这里结构图画的不严谨,Q K V按照公式来看是被unfold之后的,而作者画到立方那里去了)

- 第二步公式如下

{% asset_img 0.2.png %}

* 第三步由O生成Y

{% asset_img 0.1.png %}

具体是先reshape成原来的样子(和unfold操作相反),再经过一个1*1*1的卷积把输出的通道数改为C_o就好啦,所以Y的维数是D_qH_qW_q * C_o的。这里可以注意到Y的维数是由Qmatrix维数(D_qH_qW_q * C_k)决定的,所以可以灵活地运用在up/down_sapmle中,本文中作者将超参C_k,C_v,C_o都设成一样的了,QueryTransformC_k()作者用的是3*3*3,stride=2的nn.ConvTranspose3d。

## 实验结果

实验指标还是分割常用的DiceScore和改进的Hausdorff distance.

Hausdorff distance可以直接用下面的包(pip install 或者 conda install一下就好啦)

```python
from hausdorff import hausdorff_distance
```

DiceScore可以用下面的代码

```python
def dice_coef(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
```

我们来看看论文中作者的结果

{% asset_img 4.png %}

{% asset_img 5.png %}

{% asset_img 6.png %}

单看推理时间,确实有明显的提升。

## 代码

我们看一下代码的关键部分(注意看注释哦~)

```text
def forward(self, inputs):
    """
    :param inputs: B, C, H, W
    :return: inputs: B, Co, Hq, Wq
    """

    if self.layer_type == 'SAME' or self.layer_type == 'DOWN':
        q = self.QueryTransform(inputs)
    elif self.layer_type == 'UP':
        q = self.QueryTransform(inputs, output_size=(inputs.shape[2]*2, inputs.shape[3]*2))

    # 就是self-attention第一步里的三个并行操作得到 key matrix/value matrix/query matrix
    k = self.KeyTransform(inputs).permute(0, 2, 3, 1)
    v = self.ValueTransform(inputs).permute(0, 2, 3, 1)
    q = q.permute(0, 2, 3, 1)

    Batch, Hq, Wq = q.shape[0], q.shape[1], q.shape[2]

    #[B, H, W, N, Ck]
    k = self.split_heads(k, self.num_heads)
    v = self.split_heads(v, self.num_heads)
    q = self.split_heads(q, self.num_heads)

    # 就是self-attention第一步里的unfold操作
    k = torch.flatten(k, 0, 3)
    v = torch.flatten(v, 0, 3)
    q = torch.flatten(q, 0, 3)

    # normalize
    q = q / self._scale

    # 这就是前面提到的self-attention的第二步得到A
    A = torch.matmul(q, k.transpose(0, 1))
    A = torch.softmax(A, dim=1)
    A = self.attention_dropout(A)

    # 这就是前面提到的self-attention的第二步得到O
    O =  torch.matmul(A, v)

    # 这就是对O的fold操作
    O = O.view(Batch, Hq, Wq, v.shape[-1]*self.num_heads)
    # [B, C, Hq, Wq]
    O = O.permute(0, 3, 1, 2)
    # [B, Co, Hq, Wq]
    O = self.outputConv(O)

    return O
```

## 总结

- 简单来看,本篇就是作者将基于self-attention的全局聚合块推广到了down-sample和up-sample;
- Skip connection的concat换为sum减少参数量的同时丢失feature,所以这个residual block是否在别的数据集上有效,还是只在这个数据集上有效还待考量;
- 最近还挺多SemanticSeg和各种(花里胡哨的)attention结合的网络。