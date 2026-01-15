# -*- coding: utf-8 -*-
import yaml
import warnings

# ==============================================================================
# 1. 配置对象辅助类
# ==============================================================================

class ConfigObject:
    """
    配置辅助类。
    
    功能:
    将字典 (dict) 递归转换为对象 (object)，从而允许使用点操作符 (config.model.lr) 
    而不是字典键值对 (config['model']['lr']) 来访问参数，提高代码可读性。
    """
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                # 递归转换嵌套字典
                self.__dict__[k] = ConfigObject(v)
            else:
                self.__dict__[k] = v

    def __repr__(self):
        return str(self.__dict__)

# ==============================================================================
# 2. 解剖学映射生成逻辑 (Anatomical Mapping Generators)
# ==============================================================================

def _get_aved_mapping_32ch():
    """
    [内部函数] 生成 AVED 数据集 (32通道) 的解剖学映射配置。
    
    层级结构: 
    Level 0 (Channels): 32 个电极
    Level 1 (Regions):  8 个脑区 (Frontal, Temporal, etc.)
    Level 2 (Lobes):    3 个宏观分区 (Anterior, Posterior, etc.)
    """
    # --- 1. 定义 32 通道顺序 (AVED数据集标准) ---
    channel_map_32 = {
        0: 'Fp1', 1: 'Fp2', 2: 'F7', 3: 'F3', 4: 'Fz', 5: 'F4', 6: 'F8', 7: 'FC5', 
        8: 'FC1', 9: 'FC2', 10: 'FC6', 11: 'T7', 12: 'C3', 13: 'Cz', 14: 'C4', 15: 'T8', 
        16: 'TP9', 17: 'CP5', 18: 'CP1', 19: 'CP2', 20: 'CP6', 21: 'TP10', 22: 'P7', 
        23: 'P3', 24: 'Pz', 25: 'P4', 26: 'P8', 27: 'PO9', 28: 'O1', 29: 'Oz', 30: 'O2', 31: 'PO10'
    }
    name_to_idx = {v: k for k, v in channel_map_32.items()}
    
    # --- 2. 定义 Level 1: 中间脑区 (8个区域) ---
    regions_def = {
        'Frontal':         ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
        'FrontoCentral':   ['FC5', 'FC1', 'FC2', 'FC6'],
        'Central':         ['C3', 'Cz', 'C4'],
        'Temporal':        ['T7', 'T8'],
        'CentroParietal':  ['CP5', 'CP1', 'CP2', 'CP6'],
        'TemporoParietal': ['TP9', 'TP10'],
        'Parietal':        ['P7', 'P3', 'Pz', 'P4', 'P8'],
        'Occipital':       ['PO9', 'O1', 'Oz', 'O2', 'PO10'],
    }
    region_names = list(regions_def.keys())

    # --- 3. 定义 Level 2: 宏观脑叶 (3个分区) ---
    lobes_def = {
        'Anterior':         ['Frontal', 'FrontoCentral'],
        'Posterior':        ['Parietal', 'Occipital'],
        'Central-Temporal': ['Central', 'Temporal', 'CentroParietal', 'TemporoParietal'],
    }

    # --- 4. 生成映射索引列表 ---
    
    # Map 1: 32 Channels -> 8 Regions
    map_32_to_reg = []
    for r_name in region_names:
        # 查找当前区域包含的所有电极的索引
        indices = [name_to_idx[n] for n in regions_def[r_name] if n in name_to_idx]
        map_32_to_reg.append(sorted(indices))
        
    # Map 2: 8 Regions -> 3 Lobes
    map_reg_to_lobe = []
    for l_name in lobes_def.keys():
        # 查找当前脑叶包含的所有区域的索引
        indices = [region_names.index(n) for n in lobes_def[l_name] if n in region_names]
        map_reg_to_lobe.append(sorted(indices))
        
    all_mappings = [map_32_to_reg, map_reg_to_lobe]
    scale_node_nums = [32, len(region_names), len(lobes_def)] # [32, 8, 3]
    
    # 针对 32 通道的推荐超边数量
    hyper_num = [30, 15, 8]
    
    return all_mappings, scale_node_nums, hyper_num


def _get_dtu_mapping_64ch():
    """
    [内部函数] 生成 DTU 数据集 (64通道) 的解剖学映射配置。
    
    层级结构:
    Level 0 (Channels): 64 个电极 (从原始66通道中剔除了EXG1/2)
    Level 1 (Regions):  17 个精细脑区
    Level 2 (Lobes):    4 个宏观分区
    """
    # --- 1. 定义完整 66 通道 (DTU数据集标准) ---
    channel_map_66 = {
        0: 'Fp1', 1: 'AF7', 2: 'AF3', 3: 'F1', 4: 'F3', 5: 'F5', 6: 'F7', 7: 'FT7', 
        8: 'FC5', 9: 'FC3', 10: 'FC1', 11: 'C1', 12: 'C3', 13: 'C5', 14: 'T7', 15: 'TP7', 
        16: 'CP5', 17: 'CP3', 18: 'CP1', 19: 'P1', 20: 'P3', 21: 'P5', 22: 'P7', 23: 'P9', 
        24: 'PO7', 25: 'PO3', 26: 'O1', 27: 'Iz', 28: 'Oz', 29: 'POz', 30: 'Pz', 31: 'CPz', 
        32: 'Fpz', 33: 'Fp2', 34: 'AF8', 35: 'AF4', 36: 'AFz', 37: 'Fz', 38: 'F2', 39: 'F4', 
        40: 'F6', 41: 'F8', 42: 'FT8', 43: 'FC6', 44: 'FC4', 45: 'FC2', 46: 'FCz', 47: 'Cz', 
        48: 'C2', 49: 'C4', 50: 'C6', 51: 'T8', 52: 'TP8', 53: 'CP6', 54: 'CP4', 55: 'CP2', 
        56: 'P2', 57: 'P4', 58: 'P6', 59: 'P8', 60: 'P10', 61: 'PO8', 62: 'PO4', 63: 'O2', 
        64: 'EXG1', 65: 'EXG2' 
    }
    # 筛选前 64 个脑电通道的名称到索引映射
    name_to_idx_64 = {v: k for k, v in channel_map_66.items() if k < 64}

    # --- 2. 定义 Level 1: 精细脑区 (17个区域) ---
    regions_def = {
        'L_PreFrontal':    ['Fp1', 'AF7', 'AF3'],
        'R_PreFrontal':    ['Fp2', 'AF8', 'AF4'],
        'Mid_Frontal':     ['Fpz', 'AFz', 'Fz'],
        'L_Frontal':       ['F1', 'F3', 'F5', 'F7'],
        'R_Frontal':       ['F2', 'F4', 'F6', 'F8'],
        'L_FrontoCentral': ['FC5', 'FC3', 'FC1'],
        'R_FrontoCentral': ['FC6', 'FC4', 'FC2'],
        'Mid_Central':     ['FCz', 'Cz', 'CPz'],
        'L_Central':       ['C1', 'C3', 'C5'],
        'R_Central':       ['C2', 'C4', 'C6'],
        'L_Temporal':      ['FT7', 'T7', 'TP7'],
        'R_Temporal':      ['FT8', 'T8', 'TP8'],
        'L_Parietal':      ['CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9'],
        'R_Parietal':      ['CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10'],
        'Mid_Occipital':   ['Pz', 'POz', 'Oz', 'Iz'],
        'L_Occipital':     ['PO7', 'PO3', 'O1'],
        'R_Occipital':     ['PO8', 'PO4', 'O2'],
    }
    region_names = list(regions_def.keys())

    # --- 3. 定义 Level 2: 宏观脑叶 (4个分区) ---
    lobes_def = {
        'Frontal':         ['L_PreFrontal', 'R_PreFrontal', 'Mid_Frontal', 'L_Frontal', 'R_Frontal', 'L_FrontoCentral', 'R_FrontoCentral'],
        'CentralParietal': ['Mid_Central', 'L_Central', 'R_Central', 'L_Parietal', 'R_Parietal'],
        'Temporal':        ['L_Temporal', 'R_Temporal'],
        'Occipital':       ['Mid_Occipital', 'L_Occipital', 'R_Occipital']
    }

    # --- 4. 生成映射索引列表 ---
    
    # Map 1: 64 Channels -> 17 Regions
    map_64_to_reg = []
    for r_name in region_names:
        indices = [name_to_idx_64[n] for n in regions_def[r_name] if n in name_to_idx_64]
        map_64_to_reg.append(sorted(indices))
        
    # Map 2: 17 Regions -> 4 Lobes
    map_reg_to_lobe = []
    for l_name in lobes_def.keys():
        indices = [region_names.index(n) for n in lobes_def[l_name] if n in region_names]
        map_reg_to_lobe.append(sorted(indices))

    all_mappings = [map_64_to_reg, map_reg_to_lobe]
    scale_node_nums = [64, len(region_names), len(lobes_def)] # [64, 17, 4]
    
    # 针对 64 通道的推荐超边数量
    hyper_num = [40, 20, 10]
    
    return all_mappings, scale_node_nums, hyper_num


# ==============================================================================
# 3. 核心对外接口 (Public Interface)
# ==============================================================================

def generate_anatomical_mappings(num_channels):
    """
    [工厂函数] 根据输入的通道数，自动分发到对应的解剖学映射生成逻辑。
    
    Args:
        num_channels (int): EEG 数据集的通道数 (32 或 64)
        
    Returns:
        all_mappings (list): 包含各层级空间池化索引的列表
        scale_node_nums (list): 每一层的节点数量列表 (如 [32, 8, 3])
        hyper_num (list): 每一层建议的超边数量列表
        
    Raises:
        ValueError: 如果输入的通道数不是 32 或 64。
    """
    if num_channels == 32:
        print(">>> Info: Using AVED 32-Channel Anatomical Mapping.")
        return _get_aved_mapping_32ch()
    
    elif num_channels == 64:
        print(">>> Info: Using DTU 64-Channel Anatomical Mapping.")
        return _get_dtu_mapping_64ch()
    
    else:
        raise ValueError(
            f"Error: No anatomical mapping defined for {num_channels} channels. "
            f"Only 32 (AVED) or 64 (DTU) are currently supported."
        )

def load_config(yaml_path):
    """
    加载 YAML 配置文件并注入运行时参数。
    
    流程:
    1. 读取 YAML 文件为字典。
    2. 转换为 ConfigObject 对象。
    3. 根据 config.dataset.num_channels 自动调用 generate_anatomical_mappings。
    4. 将生成的映射、尺度节点数、超边数注入到 config 对象中。
    """
    # 1. 读取 YAML
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 2. 转换为对象
    config = ConfigObject(config_dict)
    
    # 3. 动态注入解剖学参数 (Runtime Injection)
    # 确保配置文件中必须包含 dataset.num_channels 字段
    try:
        mappings, scale_nums, hyper_nums = generate_anatomical_mappings(config.dataset.num_channels)
        
        # 将计算结果挂载到 config 对象上，供模型初始化使用
        config.all_mappings = mappings
        config.scale_node_nums = scale_nums
        config.hyper_num = hyper_nums
        
    except AttributeError:
        warnings.warn("Config file is missing 'dataset.num_channels'. Skipping anatomical mapping generation.")
    except ValueError as e:
        # 重新抛出通道数不支持的错误
        raise e
    
    return config