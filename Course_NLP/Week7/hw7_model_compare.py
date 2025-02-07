







# 对比所有模型
# 中间日志可以关掉，避免输出过多信息，
# 超参数的网络搜索，结果写入excel
import pandas as pd
df = pd.DataFrame(columns=['model_type', 'learning_rate', 'hidden_size', 
                           'batch_size', 'pooling_style', 'acc'])

