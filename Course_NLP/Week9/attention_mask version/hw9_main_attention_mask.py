# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据，包含(input_ids, attention_mask, labels)
    train_data = load_data(config["train_data_path"], config, shuffle=True)
    #加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("检测到GPU, 将模型迁移至cuda")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        logger.info("epoch %d begin" % (epoch + 1))
        model.train()
        train_loss_list = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            
            input_ids, attention_mask, labels = batch_data
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.item())
            if index % max(1, int(len(train_data) / 2)) == 0:
                logger.info("batch %d loss %f" % (index, loss.item()))
        logger.info("epoch %d average loss: %f" % (epoch + 1, np.mean(train_loss_list)))
        evaluator.eval(epoch + 1)
    final_model_path = os.path.join(config["model_path"], "epoch_%d.pth" % config["epoch"])
    torch.save(model.state_dict(), final_model_path)
    logger.info("训练完成，模型已保存到 %s" % final_model_path)
    return model, train_data

if __name__ == "__main__":
    model, train_data = main(Config)
