import json
import numpy as np
import torch
import time
from src.sim_calculation import load_embedding
from src.sim_calculation import similarity

number_of_entity = 0
number_of_training_set = 0


def load_training_set(dataset_name: str):
    """
    load training set from specified dataset
    :param dataset_name: the name of dataset
    :return: lists of triple's head, tail, relation
    """
    head = []
    tail = []
    rel = []
    with open("../dataset/" + dataset_name + "/train2id.txt", 'r', encoding="utf-8")as f:
        # print(int(f.readline()))
        global number_of_training_set
        number_of_training_set = f.readline()
        count = 0
        for line in f.readlines():
            h = int(line.strip().split(" ")[0])
            t = int(line.strip().split(" ")[1])
            r = int(line.strip().split(" ")[2])
            # print(h, r, t)
            # print(count)
            head.append(h)
            tail.append(t)
            rel.append(r)
            count += 1
        # print(count)
    print("Load training set done")
    return head, tail, rel


def load_good_and_bad_entity():
    """
    load well train entities' id and non-well train entities' id
    :return:lists of identities
    """
    good = []
    bad = []
    with open("../result/good.txt", 'r', encoding='utf-8')as f1:
        for line in f1.readlines():
            good.append(int(line.strip().split("\t")[0]))
    with open("../result/bad.txt", 'r', encoding='utf-8')as f2:
        for line in f2.readlines():
            bad.append(int(line.strip().split("\t")[0]))
    print("Got well trained and non-well trained entities")
    return good, bad


def find_similar_id(id_number: int, head, tail, rel):
    # print(id_number)
    frequency = {}
    for i in range(0, int(number_of_entity)):
        frequency[i] = 0
    # print("frequency")
    # print(frequency)
    # print("number_of_training_set:", number_of_training_set)
    count = 0
    for i in range(0, int(number_of_training_set)):
        if head[i] == id_number:
            count += 1
            target_tail = tail[i]
            target_rel = rel[i]
            # triple_list.append(str(head[i]) + ' ' + str(tail[i]) + ' ' + str(rel[i]))
            for j in range(0, int(number_of_training_set)):
                if tail[j] == target_tail and rel[j] == target_rel and head[i] != head[j]:
                    frequency[head[j]] += 1
        if tail[i] == id_number:
            count += 1
            target_head = head[i]
            target_rel = rel[i]
            # triple_list.append(str(head[i]) + ' ' + str(tail[i]) + ' ' + str(rel[i]))
            for j in range(0, int(number_of_training_set)):
                if head[j] == target_head and rel[j] == target_rel and tail[i] != tail[j]:
                    frequency[tail[j]] += 1
    # print("triple_list")
    # print(triple_list)
    frequency = sorted(frequency.items(), key=lambda item: item[1], reverse=True)
    # print(frequency)
    print("id:", id_number)
    print("show times:", count)
    lists = []
    fre = []
    for k in range(0, 10):
        lists.append(int(frequency[k][0]))
        fre.append(int(frequency[k][1]))
    return lists, fre, count


if __name__ == "__main__":
    start_time = time.time()
    dataset = "FB15K"
    result = 'TransE_epoch_500' + '.json'
    # get training set
    head, tail, rel = load_training_set(dataset)
    # get embedding we trained and the total number of entities
    ent_embedding, rel_embedding = load_embedding(result)
    number_of_entity = len(ent_embedding)
    # get the lists of good and bad entities
    good_list, bad_list = load_good_and_bad_entity()
    # find the similar entities
    with open("../result/good_sim.txt", 'w', encoding='utf-8')as f3:
        for i in range(0, len(good_list)):
            lists, fre, count = find_similar_id(good_list[i], head, tail, rel)
            for j in range(0, 10):
                sim = similarity(good_list[i], lists[j], ent_embedding)
                f3.write(
                    str(good_list[i]) + '\t' + str(lists[j]) + '\t' + str(count) + '\t' + str(fre[j]) + '\t' + str(
                        sim) + '\n')
    with open("../result/bad_sim.txt", 'w', encoding='utf-8')as f4:
        for i in range(0, len(bad_list)):
            lists, fre, count = find_similar_id(bad_list[i], head, tail, rel)
            for j in range(0, 10):
                sim = similarity(bad_list[i], lists[j], ent_embedding)
                f4.write(str(bad_list[i]) + '\t' + str(lists[j]) + '\t' + str(count) + '\t' + str(fre[j]) + '\t' + str(
                    sim) + '\n')
    # print("test number:", good_list[0])
    # find_similar_id(good_list[0], head, tail, rel)
    print("Time consumption:", time.time() - start_time, 's')
