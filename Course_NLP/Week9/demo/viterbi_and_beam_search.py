import numpy as np
import random
import copy


'''
维特比解码和beam search
'''

class Fence:
    def __init__(self, n, h):
        self.width = n
        self.height = h

    #用行列组成的list代表一个节点，每两个相邻的列的节点之间可以计算距离
    #e.g:node1 = [2,1] node2 = [3, 2]
    #为两个节点给一个固定的路径分值
    def score(self, node1, node2):
        if node1 == "start":
            return (node2[0] + node2[1] + 1) / (node2[0] * node2[1] + 1)
        assert node1[1] + 1 == node2[1] #保证两个节点处于相邻列
        mod = (node1[0] + node1[1] + node2[0] + node2[1]) % 3 + 1
        mod /= node1[0] * 4 + node1[1] * 3 + node2[0] * 2 + node2[1] * 1
        return mod

class Path:
    #定义一个路径
    #路径由数个节点组成，并且具有一个路径总分
    def __init__(self):
        self.nodes = ["start"]
        self.score = 0

    def __len__(self):
        return len(self.nodes)

def beam_search(fence, beam_size):
    width = fence.width
    height = fence.height
    starter = Path()
    beam_buffer = [starter]
    new_beam_buffer = []
    while True:
        for path in beam_buffer:
            path_length = len(path) - 1
            for h in range(height):
                node = [h, path_length]
                new_path = copy.deepcopy(path)
                new_path.score += fence.score(path.nodes[-1], node)
                new_path.nodes.append(node)
                new_beam_buffer.append(new_path)
        new_beam_buffer = sorted(new_beam_buffer, key=lambda x:x.score)
        beam_buffer = new_beam_buffer[:beam_size]
        new_beam_buffer = []
        if len(beam_buffer[0]) == width + 1:
            break
    return beam_buffer

def viterbi(fence):
    width = fence.width
    height = fence.height
    starter = Path()
    beam_buffer = [starter]
    new_beam_buffer = []
    while True:
        for h in range(height):
            path_length = len(beam_buffer[0]) - 1
            node = [h, path_length]
            node_path = []
            for path in beam_buffer:
                new_path = copy.deepcopy(path)
                new_path.score += fence.score(path.nodes[-1], node)
                new_path.nodes.append(node)
                node_path.append(new_path)
            node_path = sorted(node_path, key=lambda x:x.score)
            new_beam_buffer.append(node_path[0])
        beam_buffer = new_beam_buffer
        new_beam_buffer = []
        if len(beam_buffer[0]) == width + 1:
            break
    return sorted(beam_buffer, key=lambda x:x.score)

width = 6
height = 4
fence = Fence(width, height)
# print(fence.score([1,2], [3,3]))

beam_size = 1
res = beam_search(fence, beam_size)
for i in range(beam_size):
    print(res[i].nodes, res[i].score)
print("-----------")
res = viterbi(fence)
for path in res:
    print(path.nodes, path.score)
