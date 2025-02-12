# -*- coding: utf-8 -*-
import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data

"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {
            "LOCATION": defaultdict(int),
            "TIME": defaultdict(int),
            "PERSON": defaultdict(int),
            "ORGANIZATION": defaultdict(int)
        }
        self.model.eval()
        
        # 收集所有批次的预测和标签
        all_labels = []
        all_preds = []
        
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            
            input_ids, attention_mask, labels = batch_data  # 解包三个值
            
            with torch.no_grad():
                pred_results = self.model(input_ids, attention_mask=attention_mask)
            
            # # 检查预测结果的类型
            # self.logger.info(f"Prediction type: {type(pred_results)}")
            # if isinstance(pred_results, torch.Tensor):
            #     self.logger.info(f"Prediction shape: {pred_results.shape}")
            # elif isinstance(pred_results, list):
            #     self.logger.info(f"Prediction list length: {len(pred_results)}")
            
            # 移到CPU
            if torch.cuda.is_available():
                labels = labels.cpu()
                if not isinstance(pred_results, list):
                    pred_results = pred_results.cpu()
            
            # 收集这个批次的预测和标签
            all_labels.extend(labels.numpy())
            all_preds.extend(pred_results if isinstance(pred_results, list) else 
                            torch.argmax(pred_results, dim=-1).numpy())
        
        # 获取原始句子
        sentences = self.valid_data.dataset.sentences
        
        # 确保长度匹配
        min_len = min(len(all_labels), len(all_preds), len(sentences))
        self.write_stats(
            all_labels[:min_len],
            all_preds[:min_len],
            sentences[:min_len]
        )
        
        self.show_stats()
        return

    def write_stats(self, labels, pred_results, sentences):
        assert len(labels) == len(pred_results) == len(sentences), \
            f"长度不匹配: labels={len(labels)}, preds={len(pred_results)}, sentences={len(sentences)}"
        
        # 检查数据类型并进行相应转换
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # 处理预测结果
        if self.config["use_crf"]:
            # CRF的输出已经是列表形式，直接使用
            processed_preds = pred_results
        else:
            # 非CRF输出需要检查类型
            if isinstance(pred_results, list):
                processed_preds = pred_results
            else:
                # 如果是tensor，需要取argmax
                processed_preds = torch.argmax(pred_results, dim=-1)
                if isinstance(processed_preds, torch.Tensor):
                    processed_preds = processed_preds.cpu().numpy()
        
        # # 添加调试信息
        # self.logger.info(f"Labels type: {type(labels)}")
        # self.logger.info(f"Processed preds type: {type(processed_preds)}")
        # if isinstance(processed_preds, list):
        #     self.logger.info(f"Pred list length: {len(processed_preds)}")
        #     if processed_preds:
        #         self.logger.info(f"First pred item type: {type(processed_preds[0])}")
        
        # 对每个序列进行处理
        for true_label, pred_label, sentence in zip(labels, processed_preds, sentences):
            # 确保标签是列表格式
            true_label = true_label.tolist() if isinstance(true_label, np.ndarray) else true_label
            pred_label = pred_label.tolist() if isinstance(pred_label, np.ndarray) else pred_label
            
            # 解码得到实体
            true_entities = self.decode(sentence, true_label)
            pred_entities = self.decode(sentence, pred_label)
            
            # 统计各类实体的识别结果
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len(
                    [ent for ent in pred_entities[key] if ent in true_entities[key]]
                )
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return

    def show_stats(self):
        F1_scores = []
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, F1))
        self.logger.info("Macro-F1: %f" % np.mean(F1_scores))
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum([self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        return

    '''
    {
      "B-LOCATION": 0,
      "B-ORGANIZATION": 1,
      "B-PERSON": 2,
      "B-TIME": 3,
      "I-LOCATION": 4,
      "I-ORGANIZATION": 5,
      "I-PERSON": 6,
      "I-TIME": 7,
      "O": 8
    }
    '''
    def decode(self, sentence, labels):
        # sentence = "$" + sentence #增加一个标记在cls位置, 对齐labels
        labels = "".join([str(x) for x in labels[:len(sentence)+1]])
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results


