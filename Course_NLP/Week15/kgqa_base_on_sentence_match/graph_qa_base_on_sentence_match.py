import re
import json
import pandas
import itertools
# from py2neo import Graph
from neo4j import GraphDatabase
from collections import defaultdict


'''
使用文本匹配的方式进行知识图谱的使用
'''

class GraphQA:
    def __init__(self):
        uri = "bolt://localhost:7687"  # Neo4j默认使用bolt协议，端口通常是7687
        # 指定要使用的数据库名称
        self.database_name = "neo4j"  
        self.driver = GraphDatabase.driver(uri, auth=("neo4j", "12345678"))
        schema_path = "kg_schema.json"
        templet_path = "question_templet.xlsx"
        self.load(schema_path, templet_path)
        print("知识图谱问答系统加载完毕！\n===============")

    #加载模板
    def load(self, schema_path, templet_path):
        self.load_kg_schema(schema_path)
        self.load_question_templet(templet_path)
        return

    #加载图谱信息
    def load_kg_schema(self, path):
        with open(path, encoding="utf8") as f:
            schema = json.load(f)
        self.relation_set = set(schema["relations"])
        self.entity_set = set(schema["entitys"])
        self.label_set = set(schema["labels"])
        self.attribute_set = set(schema["attributes"])
        return

    #加载模板信息
    def load_question_templet(self, templet_path):
        dataframe = pandas.read_excel(templet_path)
        self.question_templet = []
        for index in range(len(dataframe)):
            question = dataframe["question"][index]
            cypher = dataframe["cypher"][index]
            cypher_check = dataframe["check"][index]
            answer = dataframe["answer"][index]
            self.question_templet.append([question, cypher, json.loads(cypher_check), answer])
        return


    #获取问题中谈到的实体，可以使用基于词表的方式，也可以使用NER模型
    def get_mention_entitys(self, sentence):
        return re.findall("|".join(self.entity_set), sentence)

    # 获取问题中谈到的关系，也可以使用各种文本分类模型
    def get_mention_relations(self, sentence):
        return re.findall("|".join(self.relation_set), sentence)

    # 获取问题中谈到的属性
    def get_mention_attributes(self, sentence):
        return re.findall("|".join(self.attribute_set), sentence)

    # 获取问题中谈到的标签
    def get_mention_labels(self, sentence):
        return re.findall("|".join(self.label_set), sentence)

    #对问题进行预处理，提取需要的信息
    def parse_sentence(self, sentence):
        entitys = self.get_mention_entitys(sentence)
        relations = self.get_mention_relations(sentence)
        labels = self.get_mention_labels(sentence)
        attributes = self.get_mention_attributes(sentence)
        return {"%ENT%":entitys,
                "%REL%":relations,
                "%LAB%":labels,
                "%ATT%":attributes}

    #将提取到的值分配到键上
    def decode_value_combination(self, value_combination, cypher_check):
        res = {}
        for index, (key, required_count) in enumerate(cypher_check.items()):
            if required_count == 1:
                res[key] = value_combination[index][0]
            else:
                for i in range(required_count):
                    key_num = key[:-1] + str(i) + "%"
                    res[key_num] = value_combination[index][i]
        return res

    #对于找到了超过模板中需求的实体数量的情况，需要进行排列组合
    #info:{"%ENT%":["周杰伦", "方文山"], "%REL%":["作曲"]}
    def get_combinations(self, cypher_check, info):
        slot_values = []
        for key, required_count in cypher_check.items():
            slot_values.append(itertools.combinations(info[key], required_count))
        value_combinations = itertools.product(*slot_values)
        combinations = []
        for value_combination in value_combinations:
            combinations.append(self.decode_value_combination(value_combination, cypher_check))
        return combinations

    #将带有token的模板替换成真实词
    #string:%ENT1%和%ENT2%是%REL%关系吗
    #combination: {"%ENT1%":"word1", "%ENT2%":"word2", "%REL%":"word"}
    def replace_token_in_string(self, string, combination):
        for key, value in combination.items():
            string = string.replace(key, value)
        return string

    #对于单条模板，根据抽取到的实体属性信息扩展，形成一个列表
    #info:{"%ENT%":["周杰伦", "方文山"], "%REL%":["作曲"]}
    def expand_templet(self, templet, cypher, cypher_check, info, answer):
        combinations = self.get_combinations(cypher_check, info)
        templet_cpyher_pair = []
        for combination in combinations:
            replaced_templet = self.replace_token_in_string(templet, combination)
            replaced_cypher = self.replace_token_in_string(cypher, combination)
            replaced_answer = self.replace_token_in_string(answer, combination)
            templet_cpyher_pair.append([replaced_templet, replaced_cypher, replaced_answer])
        return templet_cpyher_pair


    #验证从文本种提取到的信息是否足够填充模板，如果不足够就跳过，节省运算速度
    # 如模板：  %ENT%和%ENT%是什么关系？  这句话需要两个实体才能填充，如果问题中只有一个，该模板无法匹配
    def check_cypher_info_valid(self, info, cypher_check):
        for key, required_count in cypher_check.items():
            if len(info.get(key, [])) < required_count:
                return False
        return True

    #根据提取到的实体，关系等信息，将模板展开成待匹配的问题文本
    def expand_question_and_cypher(self, info):
        templet_cypher_pair = []
        for templet, cypher, cypher_check, answer in self.question_templet:
            if self.check_cypher_info_valid(info, cypher_check):
                templet_cypher_pair += self.expand_templet(templet, cypher, cypher_check, info, answer)
        return templet_cypher_pair

    #距离函数，文本匹配的所有方法都可以使用
    def sentence_similarity_function(self, string1, string2):
        # print("计算  %s %s"%(string1, string2))
        jaccard_distance = len(set(string1) & set(string2)) / len(set(string1) | set(string2))
        return jaccard_distance

    #通过问题匹配的方式确定匹配的cypher
    def cypher_match(self, sentence, info):
        templet_cypher_pair = self.expand_question_and_cypher(info)
        # print(templet_cypher_pair)
        result = []
        for templet, cypher, answer in templet_cypher_pair:
            score = self.sentence_similarity_function(sentence, templet)
            print(sentence, templet, score)
            result.append([templet, cypher, score, answer])
        result = sorted(result, reverse=True, key=lambda x: x[2])
        return result

    #解析结果
    def parse_result(self, graph_search_result, answer, info):
        graph_search_result = graph_search_result[0]
        #关系查找返回的结果形式较为特殊，单独处理
        if "REL" in graph_search_result:
            # neo4j驱动返回的关系对象结构与py2neo不同
            rel_value = graph_search_result["REL"]
            
            # 检查是否为字典类型
            if isinstance(rel_value, dict) and "type" in rel_value:
                graph_search_result["REL"] = rel_value["type"]
            
            # 检查是否有type属性的对象
            elif hasattr(rel_value, "type"):
                graph_search_result["REL"] = rel_value.type
            
            # 检查是否为元组或列表
            elif isinstance(rel_value, (tuple, list)):
                graph_search_result["REL"] = str(rel_value[0]) if rel_value else ""
                
            # 如果是其他类型，尝试转换为字符串
            else:
                try:
                    graph_search_result["REL"] = str(rel_value)
                except:
                    graph_search_result["REL"] = "未知关系"
        
        # 确保所有值都是字符串
        for key in graph_search_result:
            if not isinstance(graph_search_result[key], str):
                graph_search_result[key] = str(graph_search_result[key])
                
        answer = self.replace_token_in_string(answer, graph_search_result)
        return answer


    #对外提供问答接口
    def query(self, sentence):
        print("============")
        print(sentence)
        info = self.parse_sentence(sentence)    #信息抽取
        print("info:", info)
        
        # 特殊处理两个实体之间的关系查询
        if len(info["%ENT%"]) == 2 and "关系" in sentence:
            return self.get_relation_between_entities(info["%ENT%"][0], info["%ENT%"][1])
        
        templet_cypher_score = self.cypher_match(sentence, info)  #cypher匹配
        for templet, cypher, score, answer in templet_cypher_score:
            # 使用session执行查询，指定特定数据库
            with self.driver.session(database=self.database_name) as session:
                try:
                    graph_search_result = session.run(cypher).data()
                    # 最高分命中的模板不一定在图上能找到答案, 当不能找到答案时，运行下一个搜索语句, 找到答案时停止查找后面的模板
                    if graph_search_result:
                        answer = self.parse_result(graph_search_result, answer, info)
                        return answer
                except Exception as e:
                    print(f"查询执行错误: {e}")
                    continue
        return None
    
    # 专门用于查询两个实体之间关系的方法
    def get_relation_between_entities(self, entity1, entity2):
        cypher = f"""
        MATCH (a {{NAME: '{entity1}'}})-[r]->(b {{NAME: '{entity2}'}})
        RETURN type(r) as relation
        UNION
        MATCH (a {{NAME: '{entity2}'}})-[r]->(b {{NAME: '{entity1}'}})
        RETURN type(r) + '(反向)' as relation
        """
        
        with self.driver.session(database=self.database_name) as session:
            try:
                result = session.run(cypher).data()
                if result:
                    return f"{entity1}和{entity2}的关系是{result[0]['relation']}"
                else:
                    return f"没有找到{entity1}和{entity2}之间的关系"
            except Exception as e:
                print(f"关系查询错误: {e}")
                return f"查询{entity1}和{entity2}之间的关系时出错"
    
    #关闭连接
    def __del__(self):
        if hasattr(self, 'driver'):
            self.driver.close()


if __name__ == "__main__":
    graph = GraphQA()
    res = graph.query("谁导演的不能说的秘密")
    print(res)
    res = graph.query("发如雪的谱曲是谁")
    print(res)
    res = graph.query("爱在西元前的谱曲是谁")
    print(res)
    res = graph.query("周杰伦的星座是什么")
    print(res)
    res = graph.query("周杰伦的血型是什么")
    print(res)
    res = graph.query("周杰伦的身高")
    print(res)
    res = graph.query("周杰伦和淡江中学是什么关系")
    print(res)
    
    # 关闭连接
    del graph