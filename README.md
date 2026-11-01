# BEVFusionLite

从当前 OpenPCDet 提取的独立研究项目，围绕“图像–点云 BEV 自适应融合”组织代码。
项目可脱离原 OpenPCDet 目录使用；保留 `pcdet` 包名和关键参数名称，尽量兼容已有 checkpoint。
未复制数据集、预训练权重、编译产物或原仓库 Git 历史。

## 先看哪些文件

```text
BEVFusionLite/
├── train.py                         # 单 GPU 训练入口
├── test.py                          # 推理、nuScenes 评估、延迟/显存记录
├── prepare_data.py                  # 数据索引与可选 GT database
├── configs/
│   ├── nuscenes.yaml                # 公共数据配置
│   ├── centerpoint.yaml             # 点云检测入门基线
│   ├── transfusion_lidar.yaml        # BEVFusion 对应的点云基线
│   ├── bevfusion.yaml               # 原始融合基线
│   └── bevfusion_gated.yaml          # 只覆盖融合模块配置
├── pcdet/
│   ├── datasets/                    # 仅 nuScenes 与必要的数据处理
│   ├── models/detectors/bevfusion.py # 完整、显式的前向传播主线
│   ├── models/backbones_2d/fuser/    # ConvFuser 与 GatedFuser，主要修改位置
│   ├── models/view_transforms/      # 图像 -> BEV
│   ├── models/backbones_3d/         # MeanVFE 与 VoxelResBackBone8x
│   ├── models/backbones_image/      # Swin 与 FPN
│   ├── models/dense_heads/          # CenterHead 与 TransFusionHead
│   ├── ops/                        # 仅 3 组必要 CUDA 算子
│   └── utils/                      # 框、损失、稀疏卷积辅助函数
├── tools/                           # 初始化、冻结、优化器与调度器
├── scripts/                         # 环境/静态检查
└── docs/READING_GUIDE.md             # 中文阅读路线与实验约束
```

主要读 `configs/bevfusion.yaml`、`pcdet/models/detectors/bevfusion.py`、
`pcdet/models/backbones_2d/fuser/convfuser.py`、`gated_fuser.py`、`train.py`、`test.py`。
其他文件是这条主线所需的实现和运行依赖。

## 环境和安装

训练与检测仍需要 NVIDIA GPU、CUDA 工具链和 spconv。优先在 Linux / WSL2 CUDA 环境使用；
本次没有验证 Windows 原生 CUDA 编译，也没有验证完整的依赖版本组合。
可先创建独立 Python 3.10 环境，安装与你 CUDA 环境匹配的 PyTorch、torchvision、spconv，再执行：

```bash
cd BEVFusionLite
python -m pip install -r requirements.txt
python setup.py build_ext --inplace
python scripts/check_environment.py
```

`requirements.txt` 不自动选择 torch / torchvision / spconv 的 CUDA 版本。
`setup.py` 仅编译 IoU/NMS、RoIAware 和 BEV Pooling 三组算子。
默认配置不需要 SharedArray；只有手动启用 `USE_SHARED_MEMORY` 才需要另装。
使用独立环境，避免与原 OpenPCDet 的同名 `pcdet` 包混用；本项目从根目录脚本启动即可。

## 数据准备

本项目沿用上游的数据布局。传给 `--data-path` 的目录包含版本子目录：

```text
data/nuscenes/
└── v1.0-trainval/                  # mini 时替换为 v1.0-mini
    ├── samples/
    ├── sweeps/
    ├── maps/
    └── v1.0-trainval/              # 官方 JSON 标注所在目录
        ├── sample.json
        └── ...
```

```bash
python prepare_data.py --data-path data/nuscenes --version v1.0-trainval --gt-database
```

始终生成含相机信息的 infos，供三个模型复用。`--gt-database` 为 CenterPoint / TransFusion-L 的
数据库采样增强生成数据；只使用 BEVFusion 可省略该选项。
也可以传入已有的符合上述结构的数据目录，无须复制数据到本项目。
`--sweeps` 默认为 10；更改它时，训练配置中的 `MAX_SWEEPS`、`INFO_PATH`、采样数据库路径也需同步修改。
mini 适合检查流程，不能用来代替完整验证集的性能结果。

## 训练和微调

先把上游兼容的 Swin 预训练权重放在 `checkpoints/swint-nuimages-pretrained.pth`，
或修改 `configs/bevfusion.yaml` 中的 `IMAGE_BACKBONE.INIT_CFG.checkpoint`。
点云/BEVFusion 预训练权重自行准备，路径通过 `--pretrained` 指定。
权重来源可参考 [上游 BEVFusion 说明](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/guidelines_of_approaches/bevfusion.md)。

```bash
# CenterPoint：先理解点云检测流程。
python train.py --config configs/centerpoint.yaml --data-path data/nuscenes

# TransFusion-L：用于 BEVFusion 的点云分支训练或初始化。
python train.py --config configs/transfusion_lidar.yaml --data-path data/nuscenes

# 原始 BEVFusion：可从兼容的 TransFusion-L checkpoint 初始化。
python train.py --config configs/bevfusion.yaml --pretrained checkpoints/transfusion_lidar.pth

# 门控微调：从完整 BEVFusion checkpoint 加载，先只训练融合模块。
python train.py --config configs/bevfusion_gated.yaml --pretrained checkpoints/bevfusion.pth --trainable fusion
```

`--trainable` 的含义：

| 参数 | 训练范围 |
|---|---|
| `all` | 所有参数，默认 |
| `backbones-frozen` | 冻结点云 3D Backbone 与图像 Backbone；其他模块可训练 |
| `fusion` | 仅融合模块，冻结其他参数及其 BN 统计量/随机层 |

冻结检测头时保留其训练状态，使它仍能生成 target 和 loss，并将梯度传回融合模块。
`--pretrained` 允许部分加载，日志列出未加载的参数；只训练融合模块时，应使用已训练好的完整 BEVFusion
权重，并确认缺失项仅为新增门控参数。只加载点云权重并冻结整个图像分支不适合这种微调。
新融合模块保留 `conv.*` 参数名称，使用两个空间门控。权重经 Softmax 后乘 2，零初始化使初始输出
与加载同一卷积权重的 ConvFuser 一致；这是待实验验证的实现，不代表已经取得精度提升。

常用参数：`--batch-size 1 --epochs 6 --workers 4 --lr 0.0001 --amp --seed 666`。
每轮保存 checkpoint，记录 TensorBoard、配置快照与命令参数。
日志/权重默认在 `output/<配置名>/`，可用 `--output` 为不同实验指定独立目录。
本版聚焦单 GPU，未保留多机、Slurm、自动轮询 checkpoint 或分布式训练入口。
优化器及 OneCycle / cosine 调度沿用上游实现；不同总 batch size 下的结果应重新建立自己的基线。

恢复训练时使用本入口保存的 checkpoint，并保留原配置、总训练轮数和训练选项：

```bash
python train.py --config configs/bevfusion_gated.yaml --trainable fusion --resume output/bevfusion_gated/ckpt/checkpoint_epoch_2.pth
```

恢复位置为完整 epoch 边界，同时恢复优化器、AMP scaler、全局步数和随机数状态。
修改训练计划时应通过 `--pretrained` 新开实验；不承诺跨环境或不同 worker 配置的逐位一致性。
从自有完整 checkpoint 恢复/测试不再要求独立 Swin 初始化权重。

## 测试和评估

```bash
python test.py --config configs/bevfusion.yaml --checkpoint checkpoints/bevfusion.pth --batch-size 1
python test.py --config configs/bevfusion_gated.yaml --checkpoint output/bevfusion_gated/ckpt/checkpoint_epoch_6.pth --batch-size 1
```

测试严格检查模型参数完整性，防止漏加载新增模块后直接报告结果。对于门控模型，先完成训练再测试。
mini 训练与评估命令均需传 `--version v1.0-mini`；数据目录可通过 `--data-path` 覆盖。

默认在完整 val split 上执行官方 nuScenes 评估，结果写入 `output/<配置名>/eval/`：

- `result.pkl`：逐帧预测框，可用于后续可视化。
- `results_nusc.json`：nuScenes 格式预测。
- `metrics_summary.json`：官方 mAP、NDS 等指标。
- `timing.json`：GPU 同步后的耗时、显存及硬件信息。

`v1.0-test` 仅导出预测，没有本地标注可计算 mAP/NDS。
计时排除默认前 10 个 batch，覆盖模型 forward 及其中的后处理/recall，排除数据加载和 CPU->GPU 传输；
若数据不足，时间字段为 null。`--warmup` 可以调整预热数量。
batch size 为 1 时适合比较单帧延迟；大 batch 的平均每样本耗时不能作为单帧响应延迟。
显存为推理过程 PyTorch 峰值 allocated，包含模型驻留显存，不等于整个进程的显存占用。

## 验证状态

```bash
python scripts/verify_project.py
python -m unittest discover -s tests -v
```

本次已完成 Python 语法/项目内导入检查、三个命令的 `--help` 检查，以及 4 项配置/模块注册测试。
2 项需要 PyTorch 的门控梯度/冻结行为测试已提供，但当前环境缺依赖而跳过。
未执行 CUDA 编译、真实数据前向、训练、评估或权重加载验证，未生成性能结果。
本项目未包含可视化 GUI、输入退化实验脚本或最终实验报告；可在主流程验证后逐步加入。

代码阅读与消融设计见 [READING_GUIDE](docs/READING_GUIDE.md)，来源与改动见 [UPSTREAM](docs/UPSTREAM.md)。
