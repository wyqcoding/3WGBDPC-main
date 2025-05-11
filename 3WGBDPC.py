
import os
os.environ["OMP_NUM_THREADS"] = "1"
import math
import random
from munkres import Munkres
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import pdist, squareform
# 计算两点距离
from matplotlib.widgets import RectangleSelector
from sklearn.cluster import k_means
from sklearn.neighbors import KDTree, NearestNeighbors
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn import datasets, metrics
from sklearn.cluster import KMeans
def calculate_coverage(gb, total_points):
    return len(gb) / total_points if total_points > 0 else 0
perc =1

def calculate_specificity(gb):
    center = gb.mean(axis=0)
    compactness = np.mean(np.linalg.norm(gb - center, axis=1))  # 粒球内点到中心的平均距离
    return compactness



def splits(gb_list, num, splitting_method, coverage_threshold, specificity_threshold, total_points):
    gb_list_new = []
    for gb in gb_list:
        p = get_num(gb) 
        coverage = calculate_coverage(gb, total_points) 
        specificity = calculate_specificity(gb) 


        if coverage <= coverage_threshold or specificity <= specificity_threshold:
            gb_list_new.append(gb)
        else:
            gb_list_new.extend(splits_ball(gb, splitting_method))  
    return gb_list_new


def splits_ball(gb, splitting_method):
    splits_k = 2
    ball_list = []

    len_no_label = np.unique(gb, axis=0)
    if splitting_method == '2-means':
        if len_no_label.shape[0] < splits_k:
            splits_k = len_no_label.shape[0]
        kmeans = KMeans(n_clusters=splits_k, n_init=1, random_state=8)
        kmeans.fit(gb)
        label = kmeans.labels_
    else:
        return gb

    for single_label in range(0, splits_k):
        ball_list.append(gb[label == single_label, :])
    return ball_list
def draw_point(data):
    N = data.shape[0]
    plt.figure()
    plt.axis()
    for i in range(N):
        plt.scatter(data[i][0],data[i][1],s=16.,c='k')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('origin graph')
    plt.show()



def get_num(gb):
    num = gb.shape[0]
    return num


def calculate_center_and_radius(gb):
    data_no_label = gb[:,:]
    center = data_no_label.mean(axis=0)
    radius = np.max((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
    return center, radius


def gb_plot(gb_list, plt_type=0):
    plt.figure()
    plt.axis()
    for gb in gb_list:
        center, radius = calculate_center_and_radius(gb)  
        if plt_type == 0: 
            plt.plot(gb[:, 0], gb[:, 1], '.', c='k', markersize=5)
        if plt_type == 0 or plt_type == 1:  
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            plt.plot(x, y, c='r', linewidth=0.8)
        plt.plot(center[0], center[1], 'x' if plt_type == 0 else '.', color='r')
    plt.show()

# 距离
def distances(data, p):
    return ((data - p) ** 2).sum(axis=1) ** 0.5



def get_ball_quality(gb, center):
    N = gb.shape[0]
    ball_quality =  N
    mean_r = np.mean(((gb - center) **2)**0.5)
    return ball_quality, mean_r


def ball_density2(radiusAD, ball_qualitysA, mean_rs):
    N = radiusAD.shape[0]
    ball_dens2 = np.zeros(shape=N)
    for i in range(N):
        if radiusAD[i] == 0:
            ball_dens2[i] = 0
        else:
            ball_dens2[i] = ball_qualitysA[i] / (radiusAD[i] * radiusAD[i] * mean_rs[i])
    return ball_dens2


def ball_distance(centersAD):
    Y1 = pdist(centersAD)
    ball_distAD = squareform(Y1)
    return ball_distAD


def ball_min_dist(ball_distS, ball_densS):
    N3 = ball_distS.shape[0]
    ball_min_distAD = np.zeros(shape=N3)
    ball_nearestAD = np.zeros(shape=N3)
    index_ball_dens = np.argsort(-ball_densS)
    for i3, index in enumerate(index_ball_dens):
        if i3 == 0:
            continue
        index_ball_higher_dens = index_ball_dens[:i3]
        ball_min_distAD[index] = np.min([ball_distS[index, j]for j in index_ball_higher_dens])
        ball_index_near = np.argmin([ball_distS[index, j]for j in index_ball_higher_dens])
        ball_nearestAD[index] = int(index_ball_higher_dens[ball_index_near])
    ball_min_distAD[index_ball_dens[0]] = np.max(ball_min_distAD)
    if np.max(ball_min_distAD) < 1:
        ball_min_distAD = ball_min_distAD * 10
    return ball_min_distAD, ball_nearestAD# 1.15
#画图
def ball_draw_decision(ball_densS, ball_min_distS):
    Bval1_start = time.time()
    fig, ax = plt.subplots()
    N = ball_densS.shape[0]
    lst = []
    for i4 in range(N):
        ax.plot(ball_densS[i4], ball_min_distS[i4], marker='o', markersize=4.0, c='k')
        plt.xlabel('density')
        plt.ylabel('min_dist')

        # 矩形选区选择时的回调函数
    def select_callback(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        lst.append([x1, y1])

    RS = RectangleSelector(ax, select_callback,
                           drawtype='box', useblit=True,
                           button=[1, 3],  # disable middle button
                           minspanx=0, minspany=0,
                           spancoords='data',
                           interactive=True)
    # a = Annotate()
    plt.show()
    Bval1_end = time.time()
    Bval1 = Bval1_end - Bval1_start
    return lst, Bval1

#找粒球中心点
def ball_find_centers(ball_densS, ball_min_distS, lst):


    if lst and len(lst) > 0:
       print("Debug: Length of first sublist:", len(lst[0]))

       if len(lst[0]) > 0:
                ball_density_threshold = lst[0][0]
                # 在这里继续处理 ball_density_threshold 变量以及其他代码
       else:
                print("Error: First sublist is empty.")
    else:
        print("Error: Invalid list or empty list.")


    ball_density_threshold = lst[0][0]
    ball_min_dist_threshold = lst[0][1]
    centers = []
    N4 = ball_densS.shape[0]
    for i4 in range(N4):
        if ball_densS[i4] >= ball_density_threshold and ball_min_distS[i4] >= ball_min_dist_threshold:
            centers.append(i4)
    return np.array(centers)


def ball_cluster(ball_densS, ball_centers, ball_nearest, ball_min_distS,centersA):
    print("粒球中心点个数：",K1)
    if K1 == 0:  
        print('no centers')
        return
    N5 = ball_densS.shape[0]  
    circles = np.array(ball_centers)
    ball_labs = -1 * np.ones(N5).astype(int)  
    for i5, cen1 in enumerate(ball_centers):  
        ball_labs[cen1] = int(i5 + 1)  

    ball_index_density = np.argsort(-ball_densS)  
    core_num = int((1-beta)* len(gb_list))
    core_indices = ball_index_density[:core_num]
    edge_indices = ball_index_density[core_num:]
    for i5, index2 in enumerate(core_indices):  
        if ball_labs[index2] == -1:  
            ball_labs[index2] = ball_labs[int(ball_nearest[index2])]  
    edge_labels = {}  

    k=3
    for index3 in edge_indices:
        if ball_labs[index3] == -1:  
            ball_labs[index3] = -2  

            edge_labels[index3] = [] 
            for point in gb_list[index3]: 
                distances = np.linalg.norm(centersA[core_indices] - point, axis=1)  
                nearest_circle_indices = core_indices[np.argsort(distances)[:k]]  
                nearest_labels = [ball_labs[idx] for idx in nearest_circle_indices]
                nearest_labels_count = Counter(nearest_labels)
                most_common_label, most_common_count = nearest_labels_count.most_common(1)[0]
                if len(nearest_labels_count) > 1 and nearest_labels_count.most_common(2)[0][1] == \
                        nearest_labels_count.most_common(2)[1][1]:
                    distances_to_cores = distances[:k]
                    closest_core_index = np.argmin(distances_to_cores)
                    most_common_label = ball_labs[core_indices[closest_core_index]]
                edge_labels[index3].append(most_common_label)
    return ball_labs,edge_labels

def ball_draw_cluster(centersA, radiusA, ball_labs, dic_colors, gb_list, ball_centers):
    plt.figure()
    N6 = centersA.shape[0]
    for i6 in range(N6):
        for j6, point in enumerate(gb_list[i6]):
            if ball_labs[i6] == -2:
                # 若标签为-2，设置特殊颜色
                plt.plot(point[0], point[1], marker='o', markersize=4.0, color='red')
            else:
                plt.plot(point[0], point[1], marker='o', markersize=4.0, color=dic_colors[ball_labs[i6]])
    plt.plot([], [], c='red', linestyle='None', marker='.', markersize=10, label='Fringe GBs')
    plt.legend(loc='upper right',frameon=False)  # 只显示一次图例
    plt.show()

def read_labels_from_file(file_path):
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            label = float(line.strip())  # 转换为浮点数，或者保留文本格式
            labels.append(label)
    return labels
def best_map(L1, L2):
    """L1 should be the labels and L2 should be the clustering number we got"""
    Label1 = np.unique(L1)  
    nClass1 = len(Label1)  
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]

    newL2 = np.zeros(L2.shape, dtype=int)  # 以int类型初始化newL2

    for i in range(nClass2):
        if i < len(c):
            matched_indices = L2 == Label2[i]
            if np.any(matched_indices):
                new_label_idx = c[i]
                if new_label_idx < len(Label1):  # 检查索引是否在Label1的范围内
                    new_label = Label1[new_label_idx]
                    newL2[matched_indices] = new_label
    return newL2
def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])

    return missrate

def find_index_in_gb(data_point, gb):
    for i, point in enumerate(gb):
        if np.array_equal(point, data_point):
            return i
    return None  # 如果数据点不在该粒球内，则返回 None
def assign_labels_to_data(data, gb_list, ball_labs, edge_labels):
    data_labels = []
    for i, point in enumerate(data):
        for j, gb in enumerate(gb_list):
            if any((point == x).all() for x in gb):
                if ball_labs[j] == -2:
                    # 若标签为-2，按照 edge_labels 中的值给出标签
                    index_in_gb = find_index_in_gb(point, gb)
                    if index_in_gb is not None:
                        data_labels.append(edge_labels[j][index_in_gb])
                    else:
                        print("Error: Data point not found in the corresponding grain ball.")
                else:
                    data_labels.append(ball_labs[j])
                break
    return data_labels


def compute_alpha_beta(gb_list, ball_densS):

    specificities = [calculate_specificity(gb) for gb in gb_list]

   
    spec_min, spec_max = np.min(specificities), np.max(specificities)
    den_min, den_max = np.min(ball_densS), np.max(ball_densS)

    spec_norm = (specificities - spec_min) / (spec_max - spec_min + 1e-8)
    spec_norm = 1 - spec_norm  

    # 密度归一化
    den_norm = (ball_densS - den_min) / (den_max - den_min + 1e-8)


    combined_score = 0.82 * den_norm + 0.18 * spec_norm


    alpha = np.quantile(combined_score, 0.8)  # 前80%分位数
    beta = np.quantile(combined_score, 0.2)  # 前20%分位数

    return alpha, beta
if __name__ == "__main__":
    dic_colors = {0: (.8, 0, 0), 1: (0, .8, 0),
                  2: (0, 0, .8), 3: (.8, .8, 0),
                  4: (.8, 0, .8), 5: (0, .8, .8),
                  6: (0, 0, 0), 7: (0.8, 0.8, 0.8),
                  8: (0.6, 0, 0), 9: (0, 0.6, 0),
                  10: (1, 0, .8), 11: (0, 1, .8),
                  12: (1, 1, .8), 13: (0.4, 0, .8),
                  14: (0, 0.4, .8), 15: (0.4, 0.4, .8),
                  16: (1, 0.4, .8), 17: (1, 0, 1),
                  18: (1, 0, .8), 19: (.8, 0.2, 0), 20: (0, 0.7, 0),
                  21: (0.9, 0, .8), 22: (.8, .8, 0.1),
                  23: (.8, 0.5, .8), 24: (0, .1, .8),
                  25: (0.9, 0, .8), 26: (.8, .8, 0.1),
                  27: (.8, 0.5, .8), 28: (0, .1, .8),
                  29: (0, .1, .8)
                  }

    np.set_printoptions(threshold=1e16)
    file_path = []
    line_target = read_labels_from_file(file_path)

    line_target = np.array(line_target)
    data_mat = []
    start = time.time()
    data = data_mat
    total_points=data.shape[0]
    num = k*np.sqrt(data.shape[0])

    specificity_threshold = 0.8
    coverage_threshold = num / total_points  
    print("------------------",num)
    while True:
        ball_number_1 = len(gb_list)  

        gb_list = splits(gb_list, num=num, splitting_method='2-means',
                          coverage_threshold=coverage_threshold,specificity_threshold=specificity_threshold,total_points=total_points)
        ball_number_2 = len(gb_list)  
        if ball_number_1 == ball_number_2:  
            break
    centers = []
    radiuss = []
    ball_num = []
    ball_qualitys = []
    mean_rs = []
    i = 0
    for gb in gb_list:
        center, radius = calculate_center_and_radius(gb)
        ball_quality, mean_r = get_ball_quality(gb, center)
        ball_qualitys.append(ball_quality)
        mean_rs.append(mean_r)
        centers.append(center)
        radiuss.append(radius)
        ball_num.append(gb.shape[0])
    centersA = np.array(centers)
    df_centers = pd.DataFrame(centersA)
    radiusA = np.array(radiuss)
    ball_numA = np.array(ball_num)
    ball_qualitysA = np.array(ball_qualitys)
    ball_densS = ball_density2(radiusA, ball_qualitysA, mean_rs)
    index_ball_dens = np.argsort(-ball_densS)
    ball_distS = ball_distance(centersA)
    ball_min_distS, ball_nearest = ball_min_dist(ball_distS, ball_densS)
    alpha, beta = compute_alpha_beta(gb_list, ball_densS)
    start1 = time.time()
    lst, Bval1 = ball_draw_decision(ball_densS, ball_min_distS)
    end1 = time.time()
    ball_centers = ball_find_centers(ball_densS, ball_min_distS, lst)
    ball_labs,edge_labs= ball_cluster(ball_densS, ball_centers, ball_nearest, ball_min_distS,centersA)
    end = time.time()
    times = (end - start) - (end1 - start1)
    # print(edge_labs)
    print('The running time is：%s s ' % (times))
    print('Please wait for drawing clustering results......')
    print('Complete!')