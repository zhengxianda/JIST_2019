import numpy as np
import matplotlib.pyplot as plt

'''
This code is made for draw pictures by using data in /result/Complexity.txt
'''


def load_data() -> np.ndarray:
    """
    load data for complexity analysis
    items:loss, mrr_filter, accuracy, F_norm, upper(In order)
    :return: a np array saving data
    """
    with open("../result/dim_result.txt", 'r', encoding='utf-8') as f:
        f.readline()
        tests_list = []
        for line in f.readlines():
            case = []
            # dim
            case.append(int(line.strip().split(' ')[1]))
            # loss
            case.append(float(line.strip().split(' ')[4]))
            # mrr
            case.append(float(line.strip().split(' ')[5]))
            # accuracy
            case.append(float(line.strip().split(' ')[6]))
            # ent_lp_norm
            case.append(float(line.strip().split(' ')[7]))
            # rel_lp_norm
            case.append(float(line.strip().split(' ')[8]))
            # ent_F_norm
            case.append(float(line.strip().split(' ')[9]))
            # rel_F_norm
            case.append(float(line.strip().split(' ')[10]))
            # max ||h||+||r||+||t||
            case.append(float(line.strip().split(' ')[11]))
            # avg ||h||+||r||+||t||
            case.append(float(line.strip().split(' ')[12]))
            # print(case)
            tests_list.append(case)
    tests_list = np.asarray(tests_list)
    return tests_list[0:7]


def plot(data: np.ndarray):
    """
    draw pictures to show the analysis results
    :param data:
    :return: none
    """
    # print(data)
    dim = data[:, 0]
    loss = data[:, 1]
    mrr = data[:, 2]
    accuracy = data[:, 3]
    ent_lp_norm = data[:, 4]
    rel_lp_norm = data[:, 5]
    ent_F_norm = data[:, 6]
    rel_F_norm = data[:, 7]
    max = data[:, 7]
    avg = data[:, 8]
    mrr_acc = mrr + accuracy
    lp_norm = ent_lp_norm + rel_lp_norm
    F_norm = ent_F_norm + rel_F_norm

    x_axis = dim
    x_axis_name = 'dim'
    y_axis = mrr
    y_axis_name = 'mrr'

    my_x_ticks = np.arange(0, 251, 25)
    # my_y_ticks = np.arange(-2, 2, 0.3)
    plt.xticks(my_x_ticks)
    # plt.yticks(my_y_ticks)

    # plt.scatter(x=x_axis, y=y_axis, facecolor="blue")
    # plt.plot(x_axis, y_axis, 'b^-')

    # plt.plot(dim, F_norm, 'r--', label='F_norm')
    # plt.plot(dim, ent_F_norm, 'g--', label='ent_F_norm')
    # plt.plot(dim, rel_F_norm, 'b--', label='rel_F_norm')
    # plt.plot(dim, F_norm, 'ro-', dim, ent_F_norm, 'g+-', dim, rel_F_norm, 'b^-')
    # y_axis_name = 'value'

    # plt.plot(dim, ent_F_norm, 'g--', label='ent_F_norm')
    # plt.plot(dim, ent_F_norm, 'g+-')
    # y_axis_name = 'value'

    # plt.plot(dim, rel_F_norm, 'b--', label='rel_F_norm')
    # plt.plot(dim, rel_F_norm, 'b^-')
    # y_axis_name = 'value'

    # plt.plot(dim, lp_norm, 'r--', label='lp_norm')
    # plt.plot(dim, ent_lp_norm, 'g--', label='ent_lp_norm')
    # plt.plot(dim, rel_lp_norm, 'b--', label='rel_lp_norm')
    # plt.plot(dim, lp_norm, 'ro-', dim, ent_lp_norm, 'g+-', dim, rel_lp_norm, 'b^-')
    # y_axis_name = 'value'

    # plt.plot(dim, ent_lp_norm, 'g--', label='ent_lp_norm')
    # plt.plot(dim, ent_lp_norm, 'g+-')
    # y_axis_name = 'value'

    plt.plot(dim, rel_lp_norm, 'b--', label='rel_lp_norm')
    plt.plot(dim, rel_lp_norm, 'b^-')
    y_axis_name = 'value'

    # A1, B1 = optimize.curve_fit(f_1, x_axis, y_axis)[0]
    # x1 = np.asarray([0.894, 0.9])
    # x1 = x_axis
    # y1 = A1 * x1 + B1
    # plt.plot(x1, y1, "red")
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.title('TransR')
    plt.legend()
    plt.show()


def main():
    """
    main function
    :return: none
    """
    data = load_data()
    # print(data)
    plot(data)


if __name__ == '__main__':
    main()
