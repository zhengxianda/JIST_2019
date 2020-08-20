import json
import numpy as np
import torch
import time


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
    print("Load embedding done")
    return ent, rel


def load_good_and_bad_id():
    """
    find well trained entity and non-well trained entity
    :return: well trained entity list and their score
    :return: non-well trained entity list and their score
    """
    good = []
    good_avg_rank = []
    bad = []
    bad_avg_rank = []
    with open('../result/good.txt', 'r', encoding='utf-8')as f1:
        for line in f1.readlines():
            good.append(int(line.strip().split("\t")[0]))
            good_avg_rank.append(float(line.strip().split("\t")[1]))
    with open('../result/bad.txt', 'r', encoding='utf-8')as f2:
        for line in f2.readlines():
            bad.append(int(line.strip().split("\t")[0]))
            bad_avg_rank.append(float(line.strip().split("\t")[1]))
    return good, good_avg_rank, bad, bad_avg_rank


def similarity(first: int, second: int, ent_emb) -> float:
    """
    get two entity id and get their similarity
    :param first: first entity id
    :param second: second entity id
    :param ent_emb: the embedding of entity
    :return: similarity score
    """
    # print(ent_emb[first].size())
    a = torch.cosine_similarity(ent_emb[first], ent_emb[second], dim=0)
    a = float(torch.norm(a).data.numpy())
    return a


if __name__ == '__main__':
    start_time = time.time()
    # the name of result
    result = 'TransE' + '.json'
    # the name of dataset
    dataset = 'FB15K'
    entity_embedding, relation_embedding = load_embedding(result)
    good_id, good_id_avg_rank, bad_id, bad_id_avg_rank = load_good_and_bad_id()
    # print(len(good_id))
    # print(good_id)
    # print(good_id_avg_rank)
    # print(bad_id)
    # print(similarity(1, 2, entity_embedding))

    print("similarity in well trained")
    well_score = []
    with open("../result/good_sim.txt", 'w', encoding='utf-8')as f3:
        for i in range(0, len(good_id) - 1):
            for j in range(i + 1, len(good_id)):
                sim = similarity(good_id[i], good_id[j], entity_embedding)
                well_score.append(float(sim))
                # print(good_id[i], good_id[j], sim)
                f3.write(str(good_id[i]) + '\t' + str(good_id[j]) + '\t' + str(sim) + '\n')
    print("similarity in non-well trained")
    not_well_socre = []
    with open("../result/bad_sim.txt", 'w', encoding='utf-8')as f4:
        for i in range(0, len(bad_id) - 1):
            for j in range(i + 1, len(bad_id)):
                sim = similarity(bad_id[i], bad_id[j], entity_embedding)
                not_well_socre.append(float(sim))
                # print(bad_id[i], bad_id[j], sim)
                f4.write(str(bad_id[i]) + '\t' + str(bad_id[j]) + '\t' + str(sim) + '\n')
    print("arr_mean:")
    # print(well_score)
    print(np.mean(np.array(well_score)))
    print(np.mean(np.array(not_well_socre)))
    print("arr_var:")
    print(np.var(np.array(well_score)))
    print(np.var(np.array(not_well_socre)))
    print("time:")
    print(time.time() - start_time, 's')
