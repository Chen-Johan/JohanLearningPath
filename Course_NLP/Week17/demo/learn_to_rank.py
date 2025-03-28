import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, KLDivLoss

'''
learn to rank
pointwise
pairwise
listwise
模型用法示例
'''

class PointWise(nn.Module):
    def __init__(self):
        super(PointWise, self).__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 1)
        self.loss = MSELoss()

    def forward(self, user_embedding, item_embedding, label=None):
        embedding = torch.cat([user_embedding, item_embedding], dim=-1)
        predict_score = self.linear1(embedding)
        predict_score = self.linear2(predict_score).squeeze()
        if label is not None:
            return self.loss(predict_score, label)
        else:
            return predict_score

class PairWise(nn.Module):
    def __init__(self):
        super(PairWise, self).__init__()
        pass

    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine  #cos值为1，夹角为0，距离为0；cos值为0，夹角90, 距离为1

    def cosine_triplet_loss(self, a, p, n, margin=None):
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        if margin is None:
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin.squeeze()
        return torch.mean(diff[diff.gt(0)])

    def forward(self, user_embedding, positive_item_embedding, negative_item_embedding):
        loss = self.cosine_triplet_loss(user_embedding, positive_item_embedding, negative_item_embedding)
        return loss

class ListWise(nn.Module):
    def __init__(self):
        super(ListWise, self).__init__()
        self.loss = KLDivLoss(reduction="batchmean")

    def forward(self, user_embedding, list_item_embedding, label=None):
        # shape:(batch_size, emb_size) -> (batch_size, 1, emb_size)
        user_embedding = user_embedding.unsqueeze(1)
        # shape:(batch_size, item_num, emb_size) - > (batch_size, emb_size, item_num)
        list_item_embedding = list_item_embedding.transpose(1, 2)
        # (batch_size, 1, emb_size) * (batch_size, emb_size, item_num) -> (batch_size, 1, item_num)
        interact = torch.bmm(user_embedding, list_item_embedding)
        #(batch_size, 1, item_num) -> (batch_size, item_num)
        interact = interact.squeeze(1)
        if label is not None:
            interact = nn.functional.log_softmax(interact, dim=-1)
            label = nn.functional.softmax(label, dim=-1)
            return self.loss(interact, label)
        else:
            return interact


if __name__ == "__main__":
    model = PointWise()
    user_embedding = [[0.1,0.2,0.3,0.4,0.5],
                      [0.6,0.7,0.8,0.9,0.1]]
    item_embedding = [[0.3,0.6,0.8,0.0,0.3],
                      [0.6,0.1,0.6,0.4,0.3]]
    label = [1, 0]
    user_embedding = torch.FloatTensor(user_embedding)
    item_embedding = torch.FloatTensor(item_embedding)
    label = torch.LongTensor(label)
    print(model(user_embedding, item_embedding, label))



    model = PairWise()
    user_embedding = [[0.1,0.2,0.3,0.4,0.5],
                      [0.6,0.7,0.8,0.9,0.1]]
    positive_item_embedding = [[0.3,0.6,0.8,0.0,0.3],
                               [0.6,0.1,0.6,0.4,0.3]]
    negative_item_embedding = [[0.2, 0.7, 0.8, 0.0, 0.4],
                               [0.4, 0.5, 0.0, 0.3, 0.2]]
    user_embedding = torch.FloatTensor(user_embedding)
    positive_item_embedding = torch.FloatTensor(positive_item_embedding)
    negative_item_embedding = torch.FloatTensor(negative_item_embedding)
    print(model(user_embedding, positive_item_embedding, negative_item_embedding))



    model = ListWise()
    user_embedding = [[0.1,0.2,0.3,0.4,0.5],   #user1
                      [0.6,0.7,0.8,0.9,0.1]]   #user2
    # 每个user给出n个item向量
    list_item_embedding = [
        [[0.1, 0.2, 0.3, 0.4, 0.5],   #user1-item1
         [0.6, 0.7, 0.8, 0.9, 0.1],   #user1-item2
         [0.3, 0.1, 0.6, 0.1, 0.1],   #user1-item3
         [0.6, 0.9, 0.2, 0.1, 0.1]],  #user1-item4

        [[0.1, 0.2, 0.3, 0.4, 0.5],   #user2-item1
         [0.6, 0.2, 0.8, 0.3, 0.1],   #user2-item2
         [0.2, 0.4, 0.2, 0.6, 0.1],   #user2-item3
         [0.1, 0.5, 0.8, 0.1, 0.1]]   #user2-item4
    ]
    #每个item对应的得分
    label = [[0.1, 1.2, 0.4,  1],  #user1
             [3,     2,   3,  0]]  #user2
    user_embedding = torch.FloatTensor(user_embedding)
    list_item_embedding = torch.FloatTensor(list_item_embedding)
    label = torch.FloatTensor(label)
    print(model(user_embedding, list_item_embedding, label))

