# 来源与精简范围

- 来源：本地 OpenPCDet 工作区。
- 上游提交：`233f849829b6ac19afb8af8837a0246890908755`。
- 提取日期：2026-11-01。
- 上游项目：https://github.com/open-mmlab/OpenPCDet
- 许可证：项目根目录 `LICENSE`，保留源文件中已有的版权声明。

保留 nuScenes 数据读取、数据增强、体素化、三个 detector、对应 Backbone、两个 head、
LSS、融合、损失函数与必要 CUDA 算子。没有运行时依赖指向原 OpenPCDet 目录。

删除其他数据集注册、其他检测器/Backbone 注册、PFE、point head、ROI head 构建路径、
PointNet++/ROI Point Pooling/DSVT 算子、多机和 Slurm 入口、KITTI 图像数据库采样逻辑。
nuScenes 点云 GT database 采样仍保留，原有训练配置的采样策略不变。

新写单 GPU 训练/测试/数据准备入口和项目根目录相对路径的 YAML 继承加载器。
保留上游优化器/调度器；提供明确冻结模式、epoch 边界恢复和严格测试加载。
新增 GatedFuser 及继承配置，原 ConvFuser 仍可单独运行。
整理 BEVFusion forward，保留核心模块名称以便加载上游权重。

修正所保留路径中的 NumPy 旧别名、spconv 主次版本判断；旧式 checkpoint 加载显式设置 weights_only=False。
评估导出的 use_camera 元数据按实际模型设置；测试关闭点云 shuffle 并记录同步计时。
因此这里是基于原实现整理的研究代码，不能直接把上游公布指标视为本项目复现结果。
