# 阅读路线

## 第一遍：只看主线

1. `configs/bevfusion.yaml`：知道配置选中了哪些类。
2. `pcdet/models/detectors/bevfusion.py`：从 `forward()` 看双分支的执行顺序。
3. `pcdet/models/backbones_2d/fuser/convfuser.py`：理解原始拼接融合。
4. `pcdet/models/backbones_2d/fuser/gated_fuser.py`：理解空间门控。
5. `train.py`：从 batch、forward、loss、backward 读到 checkpoint。
6. `test.py`：从权重加载、预测框读到 nuScenes 评估。

第一遍暂时跳过构建器、完整 Swin、CUDA 算子及通用损失函数内部实现。

## 第二遍：跟踪 batch_dict

| 阶段 | 主要读取字段 | 主要写回字段 |
|---|---|---|
| MeanVFE | voxels、voxel_num_points | voxel_features |
| VoxelResBackBone8x | voxel_features、voxel_coords | encoded_spconv_tensor |
| HeightCompression | encoded_spconv_tensor | spatial_features |
| Swin | camera_imgs | image_features |
| GeneralizedLSSFPN | image_features | image_fpn |
| DepthLSSTransform | image_fpn、points、相机/增强矩阵 | spatial_features_img |
| ConvFuser / GatedFuser | spatial_features、spatial_features_img | spatial_features |
| BaseBEVBackbone | spatial_features | spatial_features_2d |
| TransFusionHead | spatial_features_2d、训练时的 gt_boxes | loss / final_box_dicts |

各字段以实际 forward 实现为准。运行环境就绪后在这些边界打印 tensor.shape，建立自己的形状记录表。
SparseConvTensor 的 features/indices 与 dense tensor 的 shape 应分别记录。

本配置下点云 BEV 为 256 通道，图像 BEV 为 80 通道，拼接 336 通道，融合输出 256 通道。
配置推导的空间尺寸为 180×180：点云 1440×1440 网格经 8 倍降采样；
图像 BEV 为 360×360，再经 2 倍降采样。需要在真实前向中核验尺寸和坐标方向。
`DepthLSSTransform` 最后会交换 x/y 轴，使它与点云 BEV 的布局一致。

## 第三遍：深入自己要改的模块

修改入口为 `pcdet/models/backbones_2d/fuser/`。一个新融合模块需要：

1. 接收同样的 batch_dict 字段。
2. 输出 `[B, 256, H, W]` 的 spatial_features。
3. 在该目录 `__init__.py` 注册。
4. 新建继承配置，只覆盖 FUSER。

当前 GatedFuser 学习 `[B, 2, H, W]` 权重，沿模态维做 Softmax，权重在各模态的通道维广播。
相乘时使用 `2*w` 保持中性初始化的特征幅度；Softmax 权重本身两路之和仍为 1。
`SAVE_WEIGHTS: True` 会将 detached 权重放入 batch_dict['fusion_weights']，便于后续调试/可视化。
检测头的检测损失负责训练门控，不额外提供“正确融合比例”标签。
中性初始化时首轮反传主要更新门控末层，随后前一层也可学习。

## 训练与对照

先验证完整预训练模型的推理，再跑短训练，确认损失有限、融合参数有梯度、冻结参数不变化。
最后分别训练 ConvFuser 与 GatedFuser，保持权重来源、训练样本、总 batch size、epoch、学习率、冻结策略一致。
不要只微调门控版本，却拿未做相同微调的原模型作为唯一对照。

CenterPoint 使用 CenterHead；BEVFusion 使用 TransFusionHead。
CenterPoint 用于掌握点云检测；主要融合消融比较两个 BEVFusion 配置。
可增加 TransFusion-L 作为点云基线，但需说明训练策略和网络结构差异。

建议记录：mAP、NDS、各类别 AP、参数量、同步推理时间、峰值显存，以及随机种子。
进一步比较全局门控、空间门控和参数量接近的普通卷积，区分门控方式与增加容量的收益。

## 后续鲁棒性实验的位置

图像亮度扰动放在读取图像后、归一化前；点云下采样放在体素化前。
对相同样本固定扰动种子与强度，各模型读取同一退化输入，保留完整验证集。
当前 DepthLSS 使用点云投影的稀疏深度辅助图像提升，所以减少点云会影响两个分支。
“原始输入整体退化”与“仅点云特征分支退化”是不同实验，必须写清扰动作用位置。
权重热力图可辅助解释，但权重变化不能单独证明其等价于模态可靠性或性能提升。
