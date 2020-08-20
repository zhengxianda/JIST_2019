import json
import numpy as np
import torch
import time
import torch.nn as nn

dim = 100
size_of_entity = 0
size_of_relation = 0
size_of_dataset = 0


def load_embedding(res: str):
    """
    load embedding from model
    :param res: name of dataset (string)
    :return: two tensor of embedding (two tensor)
    """
    result = json.load(open('../data/' + res, 'r', encoding='utf8'))
    # get entity embedding
    ent = result['ent_embeddings.weight']
    ent = np.array(ent)
    ent = torch.from_numpy(ent)
    # get relation embedding
    rel = result['rel_embeddings.weight']
    rel = np.array(rel)
    rel = torch.from_numpy(rel)
    global size_of_entity
    size_of_entity = len(ent)
    global size_of_relation
    size_of_relation = len(rel)
    ent_embeddings = nn.Embedding(len(ent), dim)
    rel_embeddings = nn.Embedding(len(rel), dim)
    ent_embeddings.weight.data.copy_(ent)
    rel_embeddings.weight.data.copy_(rel)
    print("Load embedding done")
    return ent_embeddings, rel_embeddings


def load_testset(dataset: str):
    """
    load test2id.txt from dataset
    :param dataset: the name of dataset (string)
    :return: the head,tail,relation of triples (three sets)
    """
    # init three sets
    head = []
    tail = []
    rel = []
    # open the test2id.txt
    with open("../dataset/" + dataset + "/test2id.txt", "r") as f1:
        # get the size of the testset
        total_number = int(f1.readline())
        global size_of_dataset
        size_of_dataset = total_number
        for line in f1.readlines():
            # get h,t,r in each lines
            h = int(line.strip().split(" ")[0])
            t = int(line.strip().split(" ")[1])
            r = int(line.strip().split(" ")[2])
            # add them into sets
            head.append(h)
            tail.append(t)
            rel.append(r)
            # print(h, t, r)
    # print("The size of dataset is", total_number)
    print("Load testset done")
    return head, tail, rel


def get_right(h: int, t: int, r: int, entity_emb, rel_emb):
    """
    left side link prediction
    get rank of (h,r,t)
    1. get loss for (h,r,t) for standard
    2. get loss for (h,i,t) , which 'i' is one of other entities
    3. if other's loss in less than standard,rank++
    :param h: head id (int)
    :param t: tail id (int)
    :param r: relation id (int)
    :param entity_emb: embeddding of enetity
    :param rel_emb: embedding of relation
    :return: the rank of (h,r,t) in left side link prediction
    """
    # init rank=1
    count = 1
    # print(h, t, r)

    # get head_ids, tail_ids, relation_ids
    # transfor them into tensor:"head", "tail" and "rel"
    headlist = np.ones(size_of_entity, dtype=np.int64)
    headlist = headlist * h
    taillist = np.arange(size_of_entity, dtype=np.int64)
    rellist = np.ones(size_of_entity, dtype=np.int64)
    rellist = rellist * r
    head = entity_emb(torch.from_numpy(headlist))
    tail = entity_emb(torch.from_numpy(taillist))
    rel = rel_emb(torch.from_numpy(rellist))
    # loss function
    # s = h + r - t
    s = torch.norm(head + rel - tail, 1, -1)
    # print(s.size())
    # print(s)
    s = s.data.numpy()
    # get (h,t,r)'s score
    standard = s[t]
    # get other's score and get rank
    for i in range(0, len(s)):
        if t != i:
            if s[i] < standard:
                count += 1
    return count


def get_left(h: int, t: int, r: int, entity_emb, rel_emb):
    """
    left side link prediction
    get rank of (h,r,t)
    1. get loss for (h,r,t) for standard
    2. get loss for (i,r,t) , which 'i' is one of other entities
    3. if other's loss in less than standard,rank++
    :param h: head id (int)
    :param t: tail id (int)
    :param r: relation id (int)
    :param entity_emb: embeddding of enetity
    :param rel_emb: embedding of relation
    :return: the rank of (h,r,t) in left side link prediction
    """
    # init rank=1
    count = 1
    # print(h, t, r)

    # get head_ids, tail_ids, relation_ids
    # transfor them into tensor:"head", "tail" and "rel"
    headlist = np.arange(size_of_entity, dtype=np.int64)
    taillist = np.ones(size_of_entity, dtype=np.int64)
    taillist = taillist * t
    rellist = np.ones(size_of_entity, dtype=np.int64)
    rellist = rellist * r
    head = entity_emb(torch.from_numpy(headlist))
    tail = entity_emb(torch.from_numpy(taillist))
    rel = rel_emb(torch.from_numpy(rellist))
    # loss function
    # s = h + r - t
    s = torch.norm(head + rel - tail, 1, -1)
    s = s.data.numpy()
    # get (h,t,r)'s score
    standard = s[h]
    # get other's score and get rank
    for i in range(0, len(s)):
        if t != i:
            if s[i] < standard:
                count += 1
    return count


if __name__ == '__main__':
    # the name of result
    result = 'TransE' + '.json'
    # the name of dataset
    dataset = 'FB15K'
    # load embedding
    entity_embedding, relation_embedding = load_embedding(result)
    # load testset
    head, tail, rel = load_testset(dataset)
    # get current time
    times = time.time()
    # save ranks and average ranks
    rank_sum = np.zeros(size_of_entity)
    rank_time = np.zeros(size_of_entity)
    rank_avg = np.zeros(size_of_entity)
    # run each case in the testset
    for i in range(0, size_of_dataset):
        h = head[i]
        t = tail[i]
        r = rel[i]
        # print(h, t, r)
        count_left = get_left(h, t, r, entity_embedding, relation_embedding)
        count_right = get_right(h, t, r, entity_embedding, relation_embedding)
        # display the progress (show index)
        if i % 1000 == 0:
            print(i)
            # print('i =', i)
            # print(count_left)
            # print(count_right)
        # print(count_left)
        # print(count_right)
        rank_sum[t] += count_left
        rank_time[t] += 1
        rank_sum[h] += count_right
        rank_time[h] += 1
    # get time consumption
    for i in range(0, len(rank_avg)):
        if rank_time[i] == 0:
            rank_avg[i] = 0
        else:
            rank_avg[i] = rank_sum[i] / rank_time[i]
    for i in range(0, len(rank_avg)):
        if 0 < rank_avg[i] < 3 and rank_time[i] > 10:
            with open("../result/good.txt", 'a', encoding='utf8') as f1:
                f1.write(str(i) + '\t' + str(rank_avg[i]) + '\n')
        if rank_avg[i] >= 1000 and rank_time[i] > 10:
            with open("../result/bad.txt", 'a', encoding='utf8') as f2:
                f2.write(str(i) + '\t' + str(rank_avg[i]) + '\n')
    print(time.time() - times, 's')
