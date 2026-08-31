YOLO_OBB训练实验指导书

# 软件环境与安装部署

1.  安装官方库 查看官方教程[https://docs.ultralytics.com/zh/quickstart/\
    ](https://docs.ultralytics.com/zh/quickstart/)![c30e2ba3-b9ae-4a86-8b18-2afcce88c025](media/image1.png){width="6.684027777777778in"
    height="3.1979166666666665in"}

选择git克隆/直接到你指定的文件夹或者直接到https://github.com/ultralytics/ultralytics下载压缩包到指定文件夹中

**提供了压缩包到此指导书的文件夹中**

![894649da-1ca0-4997-af9d-2131a862943d](media/image2.png){width="6.684027777777778in"
height="3.1979166666666665in"}

解压文件夹后在vscode中打开项目后再打开终端

![](media/image3.png){width="6.688888888888889in"
height="4.114583333333333in"}

若路径前面无(base)如![](media/image4.png){width="3.0752668416447944in"
height="0.8167377515310587in"}执行conda命令会报错如下

若无(base) 执行conda env list 也没报错就不用管了

![](media/image5.png){width="6.688888888888889in"
height="1.7951388888888888in"}

此时在开始菜单搜索anaconda并打开![](media/image6.png){width="6.688888888888889in"
height="6.206944444444445in"}

Powershell执行conda init powershell CMD执行conda init cmd.exe\
然后**重启终端 执行conda env list 下面就是存在的环境
保证接下来创建的环境的名字不重复就好了**![](media/image7.png){width="5.008767497812773in"
height="2.100181539807524in"}

然后依次执行conda create -n yolo_env python=3.10 -y
![](media/image8.png){width="6.688888888888889in"
height="7.314583333333333in"}

conda activate yolo_env 其中yolo_env这个地方换成你取的名字

![](media/image9.png){width="6.333882327209099in"
height="2.391874453193351in"}

当前面的base换成了你的环境名就行了 这个就是这个项目的环境了
**每个项目应有单独的环境**

**cd 到ultralytics-main目录路径下：**

![](media/image10.png){width="6.685416666666667in"
height="0.2743055555555556in"}

然后根据官方教程在我们刚刚创建的环境中执行**pip install -e .**
注意后面有个点啊

后面加上镜像源地址：\
**pip install -e .** -i <https://mirrors.ustc.edu.cn/pypi/simple/>

![](media/image11.png){width="6.688888888888889in"
height="3.8048611111111112in"}

![](media/image12.png){width="6.688888888888889in"
height="3.7944444444444443in"}

等待自动下载此项目的所有依赖项

![](media/image13.png){width="6.688888888888889in"
height="3.761111111111111in"}

先创建train.py后再点击右下角选择你刚刚创建并下载好依赖的环境

检测环境是否安装好了? \-\-\-\-\-\-\-\-\-\-\-\-\-\--🡪终端运行

yolo predict model=yolo26n.pt
source=\'https://ultralytics.com/images/bus.jpg\'

看这个目录中的结果是否生成

![](media/image14.png){width="6.688888888888889in"
height="5.904861111111111in"}

2.准备好数据集

新建文件夹datasets
创建images和labels目录(即存图片和标签)，将校正好的图片和标签放置在这个文件夹下

![](media/image15.png){width="6.688888888888889in"
height="1.8055555555555556in"}

3.执行数据处理文件prepare_split_and_augment.py，在项目路径下，该文件可划分数据集并且处理不符合训练的数据内容，根据自己的文件路径一键执行即可：

python prepare_split_and_augment.py \--source
\"D:/doc工作/杂/实验课/实验指导书_yolo_obb训练步骤/ultralytics-main/datasets\"
\--target \"D:/doc
工作/杂/实验课/实验指导书_yolo_obb训练步骤/ultralytics-main/datasets_new\"
\--ratio 0.25 \--seed 42 \--clean

4.修改配置文件

在如图路径下创建.yaml文件 命名为yolo_test.yaml

![](media/image16.png){width="6.688888888888889in"
height="4.270138888888889in"}

path:
\"D:/doc工作/杂/实验课/实验指导书_yolo_obb训练步骤/ultralytics-main/datasets_new\"
 *\# 你的数据集根目录*

train: images\\\\train

val: images\\\\val  *\# train/val 为 path 下的相对路径*

nc: 3  *\# 类别数*

names:

  - car

  - pedestrian

  - two-wheeler  *\# 类别名称，按顺序对应标签文件中的类别 ID*

task: obb  *\# 任务类型（旋转目标检测使用 obb）*

obb: true  *\# 是否使用旋转边界框（旋转目标检测设为 true）*

然后配置train.py文件![](media/image17.png){width="6.688888888888889in"
height="3.1284722222222223in"}

*from* pathlib *import* Path

*from* ultralytics *import* YOLO

def **main**() -\> None:

    *\# 使用绝对路径，避免在不同工作目录下运行时报找不到文件*

    root = Path(\_\_file\_\_).resolve().parent

    data_cfg = root / \"ultralytics\" / \"cfg\" / \"datasets\" /
\"yolo_test.yaml\"

    *if* not data_cfg.exists():

        *raise* FileNotFoundError(f\"未找到数据集配置文件: {data_cfg}\")

    *\# OBB 检测建议使用 -obb 预训练权重*

    model = YOLO(\"yolo11n-obb.pt\")

    model.train(

        *data*=str(data_cfg),  *\# 数据集配置文件*

        *epochs*=50,  *\# 训练轮数*

        *batch*=8,  *\# CPU 训练建议较小 batch*

        *imgsz*=640,

        *device*=\"cpu\",  *\# 强制使用 CPU*

        *workers*=2,

        *project*=str(root / \"runs\" / \"obb\"),

        *name*=\"train_exp\",

        *exist_ok*=True,

        *pretrained*=True,

        *verbose*=True,

    )

*if* \_\_name\_\_ == \"\_\_main\_\_\":

    main()

其他参数参考官网<https://docs.ultralytics.com/zh/modes/train/#augmentation-settings-and-hyperparameters>

![](media/image18.png){width="6.688888888888889in"
height="8.024305555555555in"}

启动train.py后 生成的模型在如图位置 为best.pt
epoll到设定的指或指定参数patience=xxx即在验证指标没有改善的情况下，等待xxx个epoch后提前停止训练。通过在性能停滞时停止训练，有助于防止[过拟合](https://www.ultralytics.com/glossary/overfitting)

![](media/image19.png){width="6.688888888888889in"
height="5.239583333333333in"}

# 修改YOLO OBB模型结构

![](media/image20.png){width="6.688888888888889in" height="1.7375in"}

你可以复制一份并重命名，例如 my_yolo26-obb.yaml

![](media/image21.png){width="6.688888888888889in"
height="4.963888888888889in"}

**1.编辑YAML文件**：使用文本编辑器打开，主要修改 backbone 和 head 部分。下面是一个修改了backbone中第一个卷积层输出通道数的示例（number从64改为32）：

![](media/image22.png){width="5.726830708661417in"
height="3.417374234470691in"}

**2.关键参数说明**：

> \[-1, 1, Conv, \[64, 3,
> 2\]\]：-1表示输入来自上一层；1表示该模块重复1次；Conv是模块名；\[64,
> 3, 2\]是参数（输出通道、卷积核、步长）。
>
> head部分的 Detect 模块最后的 \[nc, 1, 20,
> 4\]：nc是类别数，20是OBB的特定参数（角度分类数），4是边界框回归参数。

**使用新配置训练模型**：在训练命令中指定你修改后的YAML文件路径。

yolo train model=**my_yolo26-obb.yaml** data=your_dataset.yaml
epochs=100 imgsz=640 ......

**重要注意事项**

**维度匹配**：修改某一层的输出通道数（args中第一个数字）后，后续所有依赖该层输出的层输入通道数都需要相应调整，否则会报错。

**OBB特定层**：不要随意修改Detect模块的最后几个参数（如20和4），除非你非常清楚自己在做什么。

**复杂度与性能**：增加层数或通道数会提升模型容量，但也需要更多数据和算力，且可能降低推理速度。

** 进一步学习与调试**

> **学习结构语法**：建议先阅读官方文档中关于**模型配置文件**的章节，了解YAML文件的完整定义规则。
> <https://docs.ultralytics.com/zh/guides/model-yaml-config/>
>
> **可视化模型结构**：训练前，可以用以下代码打印模型结构，检查修改是否正确。
>
> from ultralytics import YOLO
>
> model = YOLO(\'my_yolo26-obb.yaml\')
>
> print(model.model)
>
> **从零训练**：结构修改后，通常需要从头开始训练（pretrained=False或直接使用YAML文件），而不是基于已有的预训练权重继续训练。这句话是指：**当你修改了模型的结构（例如改变了层数、通道数或添加/删除了模块），就不能再直接加载一个结构不同的预训练权重（.pt文件）来继续训练了**。

详细信息参考官方文档<https://docs.ultralytics.com/zh/>
