from .centerpoint import CenterPoint
from .transfusion import TransFusion
from .bevfusion import BevFusion

__all__ = {
    'CenterPoint': CenterPoint,
    'TransFusion': TransFusion,
    'BevFusion': BevFusion,
}

def build_detector(model_cfg, num_class, dataset):
    # 根据名称找到模型类，然后创建模型对象。
    # 这里的 __all__ 是一个字典：
    # __all__ = {
    #     'CenterPoint': CenterPoint,
    #     'TransFusion': TransFusion,
    #     'BevFusion': BevFusion,
    # }
    # 如果model_cfg.NAME是'BevFusion'
    # 则后面的(....)相当于是调用这个类，创建对象。因此整行等价于
    # BevFusion(
    #     model_cfg=model_cfg,
    #     num_class=num_class,
    #     dataset=dataset
    # )
    return __all__[model_cfg.NAME](model_cfg=model_cfg, num_class=num_class, dataset=dataset)
