import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import json

# from src.sim_calculation import load_good_and_bad_id

'''

'''
number_of_training_set = 0




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



def load_epoch_result():
    epochs = np.empty([20, 7])
    with open("../result/epoch_result.txt", 'r', encoding="utf-8")as f:
        count = 0
        f.readline()
        f.readline()
        f.readline()
        for line in f.readlines():
            # if count == 10 or count ==11:
            #     count += 1
            #     continue
            # epoch
            epochs[count][0] = int(line.strip().split(" ")[1])
            # mrr raw
            epochs[count][1] = float(line.strip().split(" ")[3])
            # mrr filter
            epochs[count][2] = float(line.strip().split(" ")[4])
            # ent_F_norm
            epochs[count][3] = float(line.strip().split(" ")[5])
            # rel_F_norm
            epochs[count][4] = float(line.strip().split(" ")[6])
            # upper
            epochs[count][5] = float(line.strip().split(" ")[7])
            # mean
            epochs[count][6] = float(line.strip().split(" ")[8])
            count += 1
    return epochs


def plot_pictures(matrix):
    epochs = matrix[0:9, 0]
    TransE_mrr_raw = matrix[0:9, 1]
    TransH_mrr_raw = matrix[10:19, 1]
    TransE_mrr_filter = matrix[0:9, 2]
    TransH_mrr_filter = matrix[10:19, 2]
    TransE_F_norm = matrix[0:9, 3]+matrix[0:9, 4]
    TransH_F_norm = matrix[10:19, 3]+matrix[10:19, 4]
    TransE_upper = matrix[0:9, 5]
    TransH_upper = matrix[10:19, 5]
    TransE_avg = matrix[0:9, 6]
    TransH_avg = matrix[10:19, 6]
    # plt.figure()
    plt.plot(epochs, TransE_mrr_filter, 's-', color='r', label="TransE_mrr")
    plt.plot(epochs, TransH_mrr_filter, 'o-', color='g', label="TransH_mrr")
    plt.xlabel("epochs")  # 横坐标名字
    plt.ylabel("MRR Value")  # 纵坐标名字
    plt.legend(loc="best")  # 图例
    plt.ylim(0)
    plt.title("MRR")
    plt.show()
    plt.plot(epochs, TransE_F_norm, 's-', color='r', label="TransE_F_norm")
    plt.plot(epochs, TransH_F_norm, 'o-', color='g', label="TransH_F_norm")
    plt.xlabel("epochs")  # 横坐标名字
    plt.ylabel("F-norm")  # 纵坐标名字
    plt.legend(loc="best")  # 图例
    plt.ylim(0)
    plt.title("F-norm")
    plt.show()
    plt.plot(epochs, TransE_upper, 's-', color='r', label="TransE max f(h,r,t)")
    plt.plot(epochs, TransH_upper, 'o-', color='g', label="TransE max f(h,r,t)")
    plt.plot(epochs, TransE_avg, 'g+-', color='b', label="TransE avg f(h,r,t)")
    plt.plot(epochs, TransH_avg, 'ro-', color='y', label="TransE avg f(h,r,t)")
    plt.xlabel("epochs")  # 横坐标名字
    plt.ylabel("f(h,r,t)")  # 纵坐标名字
    plt.legend(loc="best")  # 图例
    plt.ylim(0)
    plt.title("upper boundary")
    plt.show()


def frobenius_norm(embedding):
    # print(embedding.size())
    # a = torch.norm(embedding, dim=0)
    a = embedding ** 2
    # print(a)
    # print(a.size())
    a = torch.sum(a)
    # print(a.size())
    return np.sqrt(a.data.numpy())


def upper_boundary(ent, rel, dataset):
    h, t, r = load_training_set(dataset)
    # print(number_of_training_set)
    # print(h[0])
    # print(ent[h[0]])
    # print(r[0])
    # print(rel[r[0]])
    # print(t[0])
    # print(ent[t[0]])
    max_triple = torch.norm(ent[h[0]] + rel[r[0]] - ent[t[0]]).data.numpy()
    total = 0.0
    for i in range(1, int(number_of_training_set)):
        # print(i)
        # if i % 10000 == 0:
        #     print(i)
        a = torch.norm(ent[h[i]] + rel[r[i]] - ent[t[i]]).data.numpy()
        total += a
        if a > max_triple:
            max_triple = a
    mean = total / int(number_of_training_set)
    max_triple = float(max_triple)
    return max_triple, mean


if __name__ == '__main__':
    # good, bad = load_analyse_result()
    # plot_similar_id(good, bad)
    result = 'TransH_epoch_5000' + '.json'
    dataset = 'FB15K'
    # plot_2_norm(result)
    # print("well trained  mean:", np.mean(good[0:, 4]))
    # print("well trained variation:", np.var(good[0:, 4]))
    # print("non-well trained  mean:", np.mean(bad[0:, 4]))
    # print("non-well trained variation:", np.var(bad[0:, 4]))
    ent, rel = load_embedding(result)
    # print(ent.size())
    # print(frobenius_norm(ent))
    # print(frobenius_norm(rel))
    # print(upper_boundary(ent, rel, dataset))
    result_matrix = load_epoch_result()
    # print(result_matrix)
    plot_pictures(result_matrix)
