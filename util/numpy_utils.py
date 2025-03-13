# util/numpy_utils.py
"""
处理numpy类型转换的工具模块
"""
import logging

logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """
    递归转换所有numpy类型为Python原生类型

    Args:
        obj: 任意对象

    Returns:
        转换后的对象，所有numpy类型都被转换为Python原生类型
    """
    try:
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(item) for item in obj)
        else:
            return obj
    except ImportError:
        logger.warning("NumPy未安装，跳过类型转换")
        return obj
    except Exception as e:
        logger.error(f"类型转换时出错: {str(e)}")
        # 在出错时尝试最简单的方式：转为字符串后再转回原类型
        try:
            return eval(str(obj))
        except:
            # 如果都失败了，直接返回原对象
            return obj