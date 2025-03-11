import os
from zhipuai import ZhipuAI

#pip install zhipuai
#https://open.bigmodel.cn/ 注册获取APIKey

client = ZhipuAI(api_key=os.environ.get("zhipuApiKey")) # 填写您自己的APIKey
response = client.chat.completions.create(
    model="glm-3-turbo",  # 填写需要调用的模型名称
    messages=[
        {"role": "user", "content": "黑神话悟空好玩吗"},
    ],
)
print(response.choices[0].message.content)