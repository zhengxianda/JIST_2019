import math
import torch

number_of_training_set = 0


def lp_norm(model, method: str):
    """
    Calculate a lp-norm for a model
    :param model:
    :return:
    """
    result = 0.0
    embedding_list = []
    embedding_list.append(model['ent_embeddings.weight'])
    # embedding_list.append(model['rel_embeddings.weight'])
    if method == 'TransH':
        embedding_list.append(model['norm_vector.weight'])
    for Embedding in embedding_list:
        result += float(torch.norm(Embedding, p=2))
    return result


def lp_norm1(model, method: str):
    """
    Calculate a lp-norm for a model
    :param model:
    :return:
    """
    result = 0.0
    embedding_list = []
    # embedding_list.append(model['ent_embeddings.weight'])
    embedding_list.append(model['rel_embeddings.weight'])
    # if method == 'TransH':
    #     embedding_list.append(model['norm_vector.weight'])
    for Embedding in embedding_list:
        result += float(torch.norm(Embedding, p=2))
    return result


def l_norm(model, method: str, p=2, q=2.0):
    """
    Calculates a l-norm for a model.
    Args:
        model: model for which the norm should be calculated
        p: p-value
        q: p-value
    Returns:
    """
    result = 0.0
    tensorlist = []
    tensorlist.append(model['ent_embeddings.weight'])
    # tensorlist.append(model['rel_embeddings.weight'])
    # if method == 'TransH':
    #     tensorlist.append(model['norm_vector.weight'])
    for child in tensorlist:
        result += math.log(norm(child, p, q))
    return result


def l_norm1(model, method: str, p=2, q=2.0):
    """
    Calculates a l-norm for a model.
    Args:
        model: model for which the norm should be calculated
        p: p-value
        q: p-value
    Returns:
    """
    result = 0.0
    tensorlist = []
    # tensorlist.append(model['ent_embeddings.weight'])
    tensorlist.append(model['rel_embeddings.weight'])
    # if method == 'TransH':
    #     tensorlist.append(model['norm_vector.weight'])
    for child in tensorlist:
        result += math.log(norm(child, p, q))
    return result


def norm(module, p=2, q=2):
    """
    Calculates the l-norm of the weight matrix of a module
    Args:
        module: module for which the norm should be calculated
        p: p value for norm
        q: q value for norm
    Returns:
        norm value
    """
    reshaped = module.view(module.size(0), -1)
    return reshaped.norm(p=p, dim=1).norm(q).item()


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


def upper_boundary(model, dataset: str):
    h, t, r = load_training_set(dataset)
    ent = model['ent_embeddings.weight']
    rel = model['rel_embeddings.weight']
    max_triple = torch.norm(ent[h[0]] + rel[r[0]] - ent[t[0]]).data.numpy()
    total = 0.0
    for i in range(1, int(number_of_training_set)):
        a = 0.0
        a += torch.norm(ent[h[i]]).data.numpy()
        a += torch.norm(rel[r[i]]).data.numpy()
        a += torch.norm(ent[t[i]]).data.numpy()
        total += a
        if a > max_triple:
            max_triple = a
    mean = total / int(number_of_training_set)
    max_triple = float(max_triple)
    return max_triple, mean


if __name__ == '__main__':
    method = 'TransR'
    dim = 250
    result = method + '_' + str(dim) + '.ckpt'
    dataset = 'FB15k'
    model = torch.load('../data/' + result, map_location='cpu')
    # print(model)
    print(lp_norm(model, method), lp_norm1(model, method))
    print(l_norm(model, method), l_norm1(model, method))
    print(upper_boundary(model, dataset))
