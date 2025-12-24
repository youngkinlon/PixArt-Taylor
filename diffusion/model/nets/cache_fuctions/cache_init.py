def pixart_cache_init(model_kwargs, num_steps):
    '''
    Initialization for PixArt-alpha cache.
    '''
    cache_dic = {}

    # 核心缓存字典
    # 使用这种嵌套结构可以方便通过 cache[step][layer_idx][feature_type] 访问
    cache = {}
    # 初始化步骤 -1 (用于存储初始状态或参考帧)
    cache[-1] = {j: {} for j in range(28)}
    # 预分配采样步的缓存空间
    for i in range(num_steps + 1):  # +1 保证边界安全
        cache[i] = {}
        for j in range(28):
            # PixArt 建议缓存的内容：
            # 'sa': Self-Attention 的输出
            # 'ca': Cross-Attention 的输出 (如果是静态文本，这一部分其实可以步间复用)
            # 'mlp': MLP 层的输出
            cache[i][j] = {
            }

    cache_dic['cache'] = cache
    cache_dic['cache_counter'] = 0
    # 策略参数
    cache_dic['max_order']= 1
    cache_dic['interval'] = 3
    cache_dic['cache_counter'] = 0
    # 针对 PixArt 的特定优化：
    # 如果文本 Prompt 不变，Cross-Attention 的特征可以在一定步数内完全冻结
    current = {}
    current['num_steps'] = num_steps
    current['activated_steps'] = [0]
    return cache_dic, current