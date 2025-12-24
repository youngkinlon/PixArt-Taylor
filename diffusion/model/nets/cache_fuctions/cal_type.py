def cal_type(cache_dic,current):
    # pixart是1-20 ，而 dit是 49到 0
    ## 前两步全量计算
    first_step=(current['step']<=2)
    last_step=(current['step']>=19)
    fresh_interval=cache_dic['interval']
    if (first_step) or (cache_dic['cache_counter'] == fresh_interval - 1) or last_step:
        current['type']='full'
        cache_dic['cache_counter']= 0
        current['activated_steps'].append(current['step'])
    else:
        cache_dic['cache_counter']+=1
        current['type']='taylor'