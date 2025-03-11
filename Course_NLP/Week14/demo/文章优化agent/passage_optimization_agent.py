#coding:utf-8

import os
import json
from zhipuai import ZhipuAI

'''
利用agent思想优化文章
'''

#pip install zhipuai
#https://open.bigmodel.cn/ 注册获取APIKey
def call_large_model(prompt):
    client = ZhipuAI(api_key=os.environ.get("zhipuApiKey")) # 填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-3-turbo",  # 填写需要调用的模型名称
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    response_text = response.choices[0].message.content
    return response_text

def theme_analysis_agent(article_text):
    # 向大模型提问进行文章分析
    prompt_analysis = f"请分析并输出以下文章的主题：{article_text}"
    # 调用大模型接口，假设返回的结果是一个字典，包含结构和主题信息
    theme_analysis_result = call_large_model(prompt_analysis)
    return theme_analysis_result

def language_optimization_agent(article_text, theme_analysis_result):
    # 根据文章分析结果构建提示词
    prompt_language = f"请检查下面这篇文章中的语法错误和用词不当之处，并提出优化建议。建议要尽量简练，不超过100字。\n\n文章主题：{theme_analysis_result}\n\n文章内容：{article_text}"
    language_optimization_suggestions = call_large_model(prompt_language)
    return language_optimization_suggestions

def content_enrichment_agent(article_text, theme_analysis_result):
    # 根据文章分析结果构建提示词
    prompt_content = f"请阅读下面这篇文章，根据主题为该文章提出可以进一步扩展和丰富的内容点或改进建议，比如添加案例、引用数据等。建议要尽量简练，不超过100字。\n\n文章主题：{theme_analysis_result}\n\n文章内容：{article_text}"
    content_enrichment_suggestions = call_large_model(prompt_content)
    return content_enrichment_suggestions

def readability_evaluation_agent(article_text, theme_analysis_result):
    # 根据文章分析结果构建提示词
    prompt_readability = f"请阅读下面这篇文章，根据主题评估该文章的可读性，包括段落长度、句子复杂度等，提出一些有助于文章传播的改进建议。建议要尽量简练，不超过100字。\n\n文章主题：{theme_analysis_result}\n\n文章内容：{article_text}"
    readability_evaluation_result = call_large_model(prompt_readability)
    return readability_evaluation_result

def comprehensive_optimization_agent(article, theme_analysis_result, language_optimization_suggestions, content_enrichment_suggestions, readability_evaluation_result):
    # 合并结果的逻辑可以是将各个部分的建议整理成一个结构化的文档
    final_optimization_plan = f"请阅读下面这篇文章，以及若干个负责专项优化的agent给出的改进建议，重写这篇文章，提升文章的整体质量。\n\n文章原文:{article}\n\n文章主题分析：{theme_analysis_result}\n\n语言优化建议：{language_optimization_suggestions}\n\n内容丰富建议：{content_enrichment_suggestions}\n\n可读改进建议：{readability_evaluation_result}。\n\n优化后文章："
    final_optimization_result = call_large_model(final_optimization_plan)
    return final_optimization_result


article = """
2024年8月20日，国产游戏《黑神话：悟空》正式上线，迅速引发了全网的热议与追捧，其火爆程度令人惊叹。黑悟空之所以能如此之火，原因是多方面的。
从文化内涵来看，《黑神话：悟空》深深扎根于中国传统文化。《西游记》作为中国文学的经典之作，孙悟空更是家喻户晓的英雄形象，承载着无数国人的童年回忆和文化情感。该游戏以孙悟空为主角，让玩家能够在游戏中亲身扮演齐天大圣，体验其神通广大与英勇无畏，这种文化认同感和情感共鸣是黑悟空火爆的重要基础。它不仅仅是一款游戏，更像是一场文化的回归与盛宴，让玩家在游戏的世界里重新领略中国神话的魅力，使得传统文化以一种全新的、生动的方式呈现在大众面前。
在视觉呈现方面，黑悟空堪称一场视觉盛宴。制作团队不惜投入大量的时间和精力，运用先进的游戏制作技术，精心打造了美轮美奂的游戏画面。从细腻逼真的环境场景，到栩栩如生的角色形象，再到炫酷华丽的技能特效，每一个细节都展现出了极高的制作水准。无论是神秘奇幻的山林洞穴，还是气势恢宏的天庭宫殿，都仿佛让玩家身临其境，沉浸在一个充满想象力的神话世界之中。这种极致的视觉体验，极大地满足了玩家对于游戏画面品质的追求，也是吸引众多玩家的关键因素之一。
游戏品质上，黑悟空也达到了相当高的水平。它拥有丰富多样且极具挑战性的关卡设计，玩家需要运用智慧和技巧，不断探索、战斗，才能逐步推进游戏进程。角色的技能系统丰富且独特，玩家可以通过不同的技能组合，发挥出孙悟空的各种强大能力，增加了游戏的可玩性和策略性。同时，游戏的剧情紧凑且富有深度，在遵循原著故事框架的基础上，进行了大胆的创新和拓展，为玩家呈现了一个既熟悉又充满新鲜感的西游世界，让玩家在享受游戏乐趣的同时，也能感受到一个精彩绝伦的故事。
再者，宣传推广策略也为黑悟空的火爆添了一把柴。从 2020 年开始，制作方每年 8 月 20 日都会公开最新的实机视频，这些视频在网络上广泛传播，引发了大量关注和讨论，成功地为游戏上线预热造势。在社交媒体上，关于黑悟空的话题热度持续攀升，玩家们纷纷自发地宣传分享，形成了强大的传播效应。此外，针对海外市场，黑悟空也积极开展宣传活动，通过号召海外网友参与视频投稿、与博主合作推广等方式，有效地扩大了游戏在国际上的影响力。
《黑神话：悟空》的火爆并非偶然，而是其在文化内涵、视觉呈现、游戏品质以及宣传推广等多个方面共同发力的结果。它的成功，不仅为国产游戏树立了新的标杆，也证明了中国游戏产业在技术和创意上的巨大潜力。相信在黑悟空的带动下，未来会有更多优秀的国产游戏涌现，推动中国游戏产业不断向前发展，让中国的游戏文化在全球舞台上绽放更加耀眼的光芒。同时，黑悟空也为传统文化的传承与创新提供了新的思路和途径，让传统文化在现代社会中焕发出新的活力与生机。它不仅仅是一款游戏的成功，更是中国文化与现代科技融合发展的一个精彩范例，其影响力必将深远而持久。
"""


theme_analysis_result = theme_analysis_agent(article)
language_optimization_suggestions = language_optimization_agent(article, theme_analysis_result)
content_enrichment_suggestions = content_enrichment_agent(article, theme_analysis_result)
readability_evaluation_result = readability_evaluation_agent(article, theme_analysis_result)
final_optimization_plan = comprehensive_optimization_agent(article, theme_analysis_result, language_optimization_suggestions, content_enrichment_suggestions, readability_evaluation_result)
results = {"主题分析结果": theme_analysis_result, "语言优化建议": language_optimization_suggestions, "内容丰富建议": content_enrichment_suggestions, "可读性评价结果": readability_evaluation_result, "最终优化方案": final_optimization_plan}
#存储json文件
with open('results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"最终优化方案：{final_optimization_plan}")


