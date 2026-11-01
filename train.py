# 文件说明：单 GPU 训练入口，依次执行前向计算、检测损失、反向传播和保存检查点。
"""Single-GPU training: forward -> detection loss -> backward -> checkpoint."""
import argparse  # Python 标准库，用于解析命令行中输入的 --config、--epochs 等参数。
from pathlib import Path  # 用 Path 对象处理文件路径，方便拼接目录和获取绝对路径。
from tools.runtime import add_common_arguments  # 导入项目共用的命令行参数注册函数。


# main 是训练程序的总入口；定义函数时不会开始训练，调用它才会执行函数体。
def main():
    # 创建命令行解析器；__doc__ 是本文件顶部的说明字符串。
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)  # 加入配置路径、数据路径、batch size、worker 数量和随机种子等通用参数。
    parser.add_argument('--epochs', type=int)  # 可选：指定总训练轮数，未提供时为 None，稍后使用配置值。
    parser.add_argument('--pretrained', help='initialize matching parameters from an OpenPCDet checkpoint')  # 可选：指定用于初始化模型参数的权重路径。
    parser.add_argument('--resume', help='resume a checkpoint saved by this entrypoint')  # 可选：从本训练入口保存的检查点恢复完整训练状态。
    parser.add_argument('--trainable', choices=['all', 'backbones-frozen', 'fusion'], default='all')  # 选择全量训练、冻结两路 Backbone，或只训练融合模块。
    parser.add_argument('--amp', action='store_true')  # 命令中出现 --amp 时为 True，启用自动混合精度；否则为 False。
    parser.add_argument('--lr', type=float)  # 可选：用命令行指定的浮点学习率覆盖配置中的学习率。
    args = parser.parse_args()  # 解析命令行；例如 --epochs 6 会使 args.epochs 等于 6。
    if args.pretrained and args.resume:  # 初始化新实验和恢复旧实验是两种操作，不能同时选择。
        parser.error('--pretrained and --resume are mutually exclusive')  # 输出参数错误并退出程序。
    if (args.batch_size is not None and args.batch_size < 1) or args.workers < 0:  # batch size 必须至少为 1，数据加载进程数不能为负。
        parser.error('batch size must be positive and workers must be nonnegative')  # 参数不合法时退出。
    if args.epochs is not None and args.epochs < 1:  # 仅在用户显式指定训练轮数时检查其是否为正数。
        parser.error('--epochs must be positive')  # 拒绝零轮或负数轮训练。
    if args.lr is not None and args.lr <= 0:  # 仅在用户显式指定学习率时检查其是否大于零。
        parser.error('--lr must be positive')  # 拒绝无效的学习率。

    # 以下依赖延迟到解析参数后才导入，因此运行 --help 不要求先安装 PyTorch 等训练依赖。
    import copy  # 用于深拷贝模型配置，避免整理恢复训练信息时修改原始配置。
    import json  # 用于将命令行参数保存为 JSON 文件。
    import random  # Python 随机数模块，用于保存和恢复随机状态。
    import numpy as np  # NumPy 数值计算库；这里还需要保存和恢复它的随机状态。
    import torch  # PyTorch，提供张量、GPU 计算、自动求导及模型保存等功能。
    from tensorboardX import SummaryWriter  # 将 loss、学习率等数值写入 TensorBoard 日志。
    from tqdm import tqdm  # 为 DataLoader 添加训练进度条。
    from pcdet import __version__  # 读取当前项目的代码版本标记，保存检查点时一并记录。
    from pcdet.models import build_network, load_data_to_gpu  # 分别负责创建检测模型、将 batch 中需要的内容搬到 GPU。
    from pcdet.utils import common_utils  # 导入随机种子设置、日志创建等公共工具。
    from tools.optimization import build_optimizer, build_scheduler  # 创建更新模型参数的优化器，以及控制学习率变化的调度器。
    from tools.runtime import (build_loader, disable_image_pretrain, freeze_parameters,  # 导入数据加载、关闭额外图像预训练加载、参数冻结等辅助函数。
                               load_config, require_cuda, snapshot_config, train_mode)  # 继续导入配置读取、CUDA 检查、配置保存和训练模式设置函数。

    # 第一阶段：准备运行环境、配置、输出目录和训练数据。
    require_cuda()  # 检查 CUDA GPU 是否可用；完整运行还要求事先安装依赖并编译项目算子。
    common_utils.set_random_seed(args.seed)  # 设置 Python、NumPy 和 PyTorch 等随机种子，减少实验中的随机差异。
    cfg, output = load_config(args)  # cfg 保存模型/数据/优化配置；output 是本次实验输出目录的 Path 对象。
    if cfg.DATA_CONFIG.VERSION == 'v1.0-test':  # nuScenes 官方测试集不公开训练所需的真实标注。
        parser.error('v1.0-test has no training ground truth')  # 阻止将无公开标注的 test 版本用于训练。
    if args.resume:  # 恢复训练时，模型参数将从完整检查点加载。
        disable_image_pretrain(cfg)  # 不再单独加载 Swin 初始化权重，避免恢复时还依赖额外预训练文件。
    if args.lr is not None:  # 如果命令行提供了学习率，就采用用户指定的值。
        cfg.OPTIMIZATION.LR = args.lr  # 覆盖 YAML 中的优化器学习率配置。
    epochs = args.epochs or cfg.OPTIMIZATION.NUM_EPOCHS  # 优先使用命令行总轮数，否则使用配置中的 NUM_EPOCHS。
    ckpt_dir = output / 'ckpt'  # Path 的 / 运算符表示拼接路径，生成输出目录下的 ckpt 子目录路径。
    ckpt_dir.mkdir(exist_ok=True)  # 创建检查点目录；目录已存在时不报错。
    logger = common_utils.create_logger(output / 'train.log')  # 创建日志对象，将训练信息写入 train.log 并输出到终端。
    snapshot_config(cfg, output / 'config.yaml')  # 保存本次读取并覆盖后的配置，方便日后核对实验条件。
    (output / 'arguments.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')  # 将 args 转为字典，再以缩进 JSON 保存命令行参数。
    dataset, loader = build_loader(cfg, args, logger, training=True)  # dataset 负责读/处理单个样本；loader 负责组织批次，training=True 启用训练数据流程。

    # 第二阶段：创建模型、加载初始权重，并确定哪些参数参与训练。
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), dataset).cuda()  # 根据 MODEL.NAME 创建模型，传入类别数和数据属性，再将模型搬到 GPU。
    if args.pretrained:  # 如果指定初始化权重，则加载与当前网络名称和形状匹配的参数。
        model.load_params_from_file(str(Path(args.pretrained).resolve()), logger, to_cpu=True)  # 权重文件先读到 CPU，再复制到模型；日志记录未匹配的参数。
    frozen = freeze_parameters(model, args.trainable)  # 按训练策略关闭部分参数的梯度，返回被冻结的模块名称列表。
    logger.info('Frozen modules: %s', frozen)  # 记录本次具体冻结了哪些模块。
    logger.info('Trainable parameters: %d / %d',  # 日志中的两个 %d 分别填入可训练参数数量与总参数数量。
                sum(p.numel() for p in model.parameters() if p.requires_grad),  # 遍历参数 p，仅累加允许求梯度的参数元素数；numel() 表示元素数量。
                sum(p.numel() for p in model.parameters()))  # 累加全部参数元素数，完成这次 logger.info 调用。
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)  # 创建优化器，用梯度调整模型中参与训练的参数。
    scheduler, _ = build_scheduler(optimizer, len(loader), epochs, -1, cfg.OPTIMIZATION)  # 根据每轮批次数和总轮数创建调度器；_ 表示忽略第二个返回值。
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)  # AMP 开启时缩放梯度以减少浮点下溢；关闭时按普通精度流程工作。
    start_epoch = step = 0  # 新实验从第 0 轮、第 0 次全局迭代开始；一次 batch 训练记作一次 step。

    # 第三阶段：记录训练条件；恢复旧实验时，先确认条件与检查点一致。
    model_signature = copy.deepcopy(dict(cfg.MODEL))  # 深拷贝模型配置，整理用于恢复一致性检查的信息。
    if 'IMAGE_BACKBONE' in model_signature:  # BEVFusion 有图像 Backbone，而纯点云模型可能没有。
        model_signature['IMAGE_BACKBONE'].pop('INIT_CFG', None)  # 排除仅用于初始化的设置；键不存在时返回 None，不报错。
    signature = dict(epochs=epochs, batches=len(loader), trainable=args.trainable,  # 记录总轮数、每轮批次数及冻结策略；它是设置记录，不是加密签名。
                     optimization=dict(cfg.OPTIMIZATION), model=model_signature,  # 记录优化器配置和排除初始化设置后的模型配置。
                     dataset=dict(cfg.DATA_CONFIG), batch_size=loader.batch_size, amp=args.amp,  # 记录数据配置、实际 batch size 和混合精度开关。
                     seed=args.seed, workers=args.workers)  # 记录随机种子与数据加载进程数，完成训练条件字典。
    if args.resume:  # 只有显式提供 --resume 才恢复旧实验的完整状态。
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)  # 从 CPU 读取包含模型、优化器、随机状态等内容的检查点字典。
        if checkpoint.get('run_signature') != signature:  # 比较旧实验与当前实验的关键训练条件。
            raise ValueError('Resume settings differ. Use --pretrained for a new schedule.')  # 条件不一致则停止；改变计划时应使用权重初始化方式新开实验。
        model._load_state_dict(checkpoint['model_state'], strict=True)  # 严格恢复模型参数和缓冲区，并保留项目对 spconv 权重格式的适配处理。
        optimizer.load_state_dict(checkpoint['optimizer_state'])  # 恢复优化器状态，例如 Adam 保存的梯度一阶/二阶矩估计。
        scaler.load_state_dict(checkpoint['scaler_state'])  # 恢复 AMP 梯度缩放器状态。
        start_epoch, step = checkpoint['epoch'], checkpoint['it']  # 恢复已完成的轮数和全局迭代次数，从下一轮继续。
        random.setstate(checkpoint['rng']['python'])  # 恢复 Python 随机数生成器的状态。
        np.random.set_state(checkpoint['rng']['numpy'])  # 恢复 NumPy 随机数生成器的状态。
        torch.set_rng_state(checkpoint['rng']['torch'])  # 恢复 PyTorch CPU 随机数生成器的状态。
        torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])  # 恢复各 CUDA 设备随机数生成器的状态；跨环境不保证逐位一致。

    # 第四阶段：训练主循环；先遍历 epoch，再遍历该轮的每一个 batch。
    with SummaryWriter(str(output / 'tensorboard')) as writer:  # 打开 TensorBoard 日志写入器；离开 with 时自动关闭。
        for epoch in range(start_epoch, epochs):  # epoch 从 start_epoch 取到 epochs-1；新实验从零开始计数。
            train_mode(model, frozen)  # 设置训练模式，并固定冻结模块的 BN 统计量/随机层；检测头仍保留计算损失所需的训练状态。
            progress = tqdm(loader, desc=f'epoch {epoch + 1}/{epochs}')  # 为本轮 DataLoader 包装进度条；显示轮数时加 1，便于阅读。
            for batch in progress:  # 每次得到一个 batch 字典，其中包括体素、图像、标定、真实框等当前模型需要的数据。
                scheduler.step(step, epoch)  # 根据当前全局迭代和轮数调整学习率；此处调用的是本项目配置使用的调度器。
                optimizer.zero_grad()  # 清除上一次迭代的梯度，避免本次反传时意外累加旧梯度。
                load_data_to_gpu(batch)  # 将 batch 内需要计算的数组/张量搬到 GPU，元信息等字段保留原形式。
                with torch.cuda.amp.autocast(enabled=args.amp):  # AMP 开启时，为适合的运算自动选择较低精度；关闭时不启用自动混合精度。
                    result, scalars, _ = model(batch)  # 调用模型的 forward；返回损失字典、用于日志的数值字典，以及此处忽略的显示信息。
                    loss = result['loss'].mean()  # 从结果中取检测总损失并求均值，得到可执行反向传播的单个标量。
                if not torch.isfinite(loss):  # 检查损失是否出现 NaN 或正负无穷，避免带着异常数值继续训练。
                    raise FloatingPointError(f'Nonfinite loss at step {step}: {loss.item()}')  # 遇到异常损失就停止，并报告迭代位置及损失值。
                scaler.scale(loss).backward()  # 对损失反向传播，计算可训练参数的梯度；AMP 开启时先缩放损失。
                scaler.unscale_(optimizer)  # 在梯度裁剪前撤销梯度缩放，使裁剪使用实际梯度大小。
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIMIZATION.GRAD_NORM_CLIP)  # 限制梯度的整体范数，缓解梯度过大造成的不稳定。
                scaler.step(optimizer)  # 执行参数更新；AMP 开启且检测到异常梯度时，可以跳过本次更新。
                scaler.update()  # 更新 AMP 的缩放因子，供后续迭代使用。
                model.update_global_step()  # 更新模型内部保存的全局步数计数器。
                step += 1  # 更新训练脚本的迭代计数；即使 AMP 跳过一次更新，这里仍记录已处理一个 batch。
                writer.add_scalar('train/loss', loss.item(), step)  # 将张量损失转为 Python 数值，记录到当前 step 的 loss 曲线。
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], step)  # 记录第一个优化器参数组当前的学习率。
                for key, value in scalars.items():  # 遍历模型提供的其他日志数值，例如分类损失、框回归损失等。
                    if isinstance(value, (int, float)):  # 只记录普通整数或浮点数，跳过这里未处理的其他类型。
                        writer.add_scalar(f'train/{key}', value, step)  # 为每个日志指标写入独立的 TensorBoard 曲线。
                progress.set_postfix(loss=f'{loss.item():.4f}')  # 在进度条末尾显示当前 batch 的损失，保留四位小数。

            # 第五阶段：本轮所有 batch 处理完成后，保存能从 epoch 边界恢复的训练状态。
            checkpoint = dict(  # 创建检查点字典；它包含的不只是模型权重。
                model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),  # 保存模型参数/缓冲区及优化器状态。
                scaler_state=scaler.state_dict(), epoch=epoch + 1, it=step,  # 保存 AMP 状态、已完成轮数和全局迭代次数。
                version=__version__, run_signature=signature,  # 保存代码版本标记和本次训练条件，便于追溯与恢复检查。
                rng=dict(python=random.getstate(), numpy=np.random.get_state(),  # 开始保存随机状态：Python 和 NumPy 两部分。
                         torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()),  # 继续保存 PyTorch CPU 和各 CUDA 设备随机状态，结束 rng 字典。
            )  # 结束整个检查点字典的构建。
            destination = ckpt_dir / f'checkpoint_epoch_{epoch + 1}.pth'  # 为当前轮生成正式检查点路径，例如 checkpoint_epoch_1.pth。
            temporary = destination.with_suffix('.tmp')  # 先生成同目录临时路径，避免中断时留下未写完的正式 .pth 文件。
            torch.save(checkpoint, temporary)  # 将完整检查点序列化写入临时文件。
            temporary.replace(destination)  # 临时文件写入成功后再替换为正式文件名。
            logger.info('Saved %s (step %d)', destination, step)  # 记录检查点保存位置及对应的全局迭代次数。


if __name__ == '__main__':  # 只有直接运行 python train.py 时进入；被其他文件 import 时不自动启动训练。
    main()  # 调用训练总入口；如果命令中有 --help，解析器会显示帮助并提前退出。
