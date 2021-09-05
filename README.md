- 👋 Hi, I’m @tanjiankk
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...

<!---
tanjiankk/tanjiankk is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches
import math
from sympy import *
np.seterr(divide='ignore',invalid='ignore')

#计算距离
def distance_jk(x_j, y_j, x_k, y_k):
    return abs(x_j-x_k)+abs(y_j-y_k)

def dict_all(xy_array, num):
    dict_all_list = []
    for j in range(num):
        dict_step_list = []
        for k in range(num):
            if j == k:
                dict_jk = 0
            else:
                dict_jk = distance_jk(xy_array[j:j+1,0:1][0][0], xy_array[j:j+1,1:2][0][0], 
                                      xy_array[k:k+1,0:1][0][0], xy_array[k:k+1,1:2][0][0])
            dict_step_list.append(dict_jk)
        dict_all_list.append(dict_step_list)
    return dict_all_list

#最物料搬运费用
def Objective_function_MH(distance_array, frequcy_array, cost_array):
    MH_cost = np.sum(distance_array*frequcy_array*cost_array)
    return MH_cost

def Objective_function_SR(xy_array, num):
    si_sum = np.sum(np.array([xy_array[i:i+1,2:3]*xy_array[i:i+1,-1] for i in range(num)]))
    SR_area = (max(xy_array[:,0:1])[0]-min(xy_array[:,0:1])[0])*(max(xy_array[:,1:2])[0]-min(xy_array[:,1:2])[0])
    SR_values = si_sum/SR_area
    return SR_values

#启发式布局更新xy随机生成函数
def Random_xy(H, K, n, H_oj, K_oj, lw_new, xylw_new):
    #l 设备长度， w设备宽度，H场地长度， K场地宽度， H_oj边距, K_oj边距, lw_new需要更新设备长宽， xylw_new设备信息
    x_list = []
    y_list = []
    while(len(x_list)<n):
        x = random.uniform(H_oj+lw_new[2]/2, H-H_oj-lw_new[2]/2)
        y = random.uniform(K_oj+lw_new[3]/2, K-K_oj-lw_new[3]/2)
        condition = []
        for xi,yi,li,wi in xylw_new:
            condition1 = (x<xi-li/2)|(x>xi+li/2)
            condition2 = (y<yi-wi/2)|(y>yi+wi/2) 
            condition3 = condition1 | condition2
            condition.append(condition3)
        if all(condition)&(len(x_list)<n):
            x_list.append(x)
            y_list.append(y)
    random_xy = np.hstack((np.array(x_list).reshape(n,1),np.array(y_list).reshape(n,1)))
    return random_xy
    
 #计算四周嵌入面积
def Embedded_around(H, K, H_oj, K_oj, xylw_new):
    #先判断是否嵌入边界
    embed_area_list = []
    large_rect_x1 = H_oj
    large_rect_y1 = K - K_oj
    large_rect_x2 = H - H_oj
    large_rect_y2 = K_oj
    x1, x2 = min(large_rect_x1, H - H_oj), max(large_rect_x1, H - H_oj)
    y1, y2 = min(K - K_oj, K_oj), max(K - K_oj, K_oj)
    for index in range(len(xylw_new)):
        small_rect_x1 = xylw_new[index][0] - xylw_new[index][2]/2
        small_rect_y1 = xylw_new[index][1] + xylw_new[index][3]/2
        small_rect_x2 = xylw_new[index][0] + xylw_new[index][2]/2
        small_rect_y2 = xylw_new[index][1] - xylw_new[index][3]/2
        x3, x4 = min(small_rect_x1, small_rect_x2), max(small_rect_x1, small_rect_x2)
        y3, y4 = min(small_rect_y1, small_rect_y2), max(small_rect_y1, small_rect_y2)
        
        temp_x1 = max(x1, x3)
        temp_x2 = min(x2, x4)
        temp_y1 = max(y1, y3)
        temp_y2 = min(y2, y4)
        
        if temp_x2-temp_x1<0 or temp_y2-temp_y1<0:
            area = 0
        else:
            area = (temp_x2-temp_x1)*(temp_y2-temp_y1)
        embed_area = xylw_new[index][2]*xylw_new[index][3] - area
        embed_area_list.append(embed_area)
    return embed_area_list

#     计算设备与设备嵌入面积
def Embedding_device(xylw_new):
    area_list = []
#     embedding_list = []
    for i in range(len(xylw_new)):
        area_step = []
        rect1_x1 = xylw_new[i:i+1,0:1] - xylw_new[i:i+1,2:3]/2
        rect1_y1 = xylw_new[i:i+1,1:2] - xylw_new[i:i+1,3:4]/2
        rect1_x2 = xylw_new[i:i+1,0:1] + xylw_new[i:i+1,2:3]/2
        rect1_y2 = xylw_new[i:i+1,1:2] + xylw_new[i:i+1,3:4]/2
        x1, x2 = min(rect1_x1, rect1_x2), max(rect1_x1, rect1_x2)
        y1, y2 = min(rect1_y1, rect1_y2), max(rect1_y1, rect1_y2)
        for j in range(len(xylw_new)):
            if i != j:
                rect2_x1 = xylw_new[j:j+1,0:1] - xylw_new[j:j+1,2:3]/2
                rect2_y1 = xylw_new[j:j+1,1:2] - xylw_new[j:j+1,3:4]/2
                rect2_x2 = xylw_new[j:j+1,0:1] + xylw_new[j:j+1,2:3]/2
                rect2_y2 = xylw_new[j:j+1,1:2] + xylw_new[j:j+1,3:4]/2
                
                x3, x4 = min(rect2_x1, rect2_x2), max(rect2_x1, rect2_x2)
                y3, y4 = min(rect2_y1, rect2_y2), max(rect2_y1, rect2_y2)
                
                temp_x1 = max(x1, x3)
                temp_x2 = min(x2, x4)
                temp_y1 = max(y1, y3)
                temp_y2 = min(y2, y4)
                
                if temp_x2-temp_x1<0 or temp_y2-temp_y1<0:
                    area = 0
                else:
                    area = float((temp_x2-temp_x1)*(temp_y2-temp_y1))
            if i == j:
                area = 0
            area_step.append(area)
        area_list.append(area_step)
    return area_list

#定义相对物料搬运费
def Material_charges(distance_array, frequcy_array, cost_array):
    MC_list = []
    for index in range(len(frequcy_array)):
        MC_1 = np.sum(distance_array[index]*frequcy_array[index]*cost_array[index])
        MC_2 = np.sum(frequcy_array[index]*cost_array[index])
        
        MC = MC_1/MC_2
        MC_list.append(MC)
    return MC_list

#定义弹性势能计算公式
def Elastic_energy(H, K, H_oj, K_oj, xy_array):
    device_area = Embedding_device(xy_array)
    D_ij_list = []
    D_iF_list = []
    fun_index_array = np.ones((len(xy_array), len(xy_array)))*4
    for i in range(len(xy_array)):
#         for j in range(i+1,len(xy_array),1):
        D_ij_step = []
        for j in range(len(xy_array)):
            if i != j:
                if device_area[i][j] >= 0:
                    top_move = (xy_array[i][3]+xy_array[j][3])/2-(xy_array[i][1]-xy_array[j][1])+K_oj
                    right_move = (xy_array[i][2]+xy_array[j][2])/2-(xy_array[i][0]-xy_array[j][0])+H_oj
                    bottom_move = (xy_array[i][3]+xy_array[j][3])/2-(xy_array[j][1]-xy_array[i][1])+K_oj
                    left_move = (xy_array[i][2]+xy_array[j][2])/2-(xy_array[j][0]-xy_array[i][0])+H_oj
                    values = np.array([top_move, right_move, bottom_move, left_move])
                    D_ij = min(top_move, right_move, bottom_move, left_move)
                    fun_index = np.where(values == D_ij)[0][0]
#                     print(fun_index)
#                     print(values)
                    fun_index_array[i][j] = fun_index
                else:
                    D_ij = 0
            else:
                D_ij = 0
            D_ij_step.append(D_ij)
        D_ij_list.append(float(np.sum(np.array(D_ij_step))))
        
    enbedding_area = Embedded_around(H, K, H_oj, K_oj, xy_array)
    x_f = H/2
    y_f = K/2
    fun_f_list = []
    for device_index in range(len(enbedding_area)):
        try:
            if (enbedding_area[device_index] <= 0.5) and (enbedding_area[device_index] > 0):
                D_iF = 0
                fun_index_f = 4
                enbedding_area[device_index] == 0
            elif enbedding_area[device_index] > 0.5:
                rect = xy_array[device_index]
                condition1 = rect[0]+rect[2]/2>=H-H_oj     #右边
                condition2 = rect[0]-rect[2]/2<=H_oj       #左边
                condition3 = rect[1]+rect[3]/2>=K-K_oj     #上边
                condition4 = rect[1]-rect[3]/2<=K_oj       #下边
                condition = np.array([condition1, condition2, condition3, condition4])
                if len(np.where(condition == True)[0]) == 1:
                    if condition[0] or condition[1]:
                        D_iF = np.sqrt((abs(xy_array[device_index][0]-x_f)+xy_array[device_index][2]/2-H/2)**2+xy_array[device_index][3]**2)
                        fun_index_f = 0
                    elif condition[2] or condition[3]:
                        D_iF = np.sqrt((abs(xy_array[device_index][1]-y_f)+xy_array[device_index][3]/2-K/2)**2+xy_array[device_index][2]**2)
                        fun_index_f = 1
                elif len(np.where(condition == True)[0]) == 2:
                    D_iF = np.sqrt((abs(xy_array[device_index][1]-y_f)+xy_array[device_index][3]/2-K/2)**2+\
                                (abs(xy_array[device_index][0]-x_f)+xy_array[device_index][2]/2-H/2)**2)
                    fun_index_f = 2
            else:
                D_iF = 0
                fun_index_f = 4
            D_iF_list.append(D_iF)
            fun_f_list.append(fun_index_f)
        except:
            print((rect),enbedding_area[device_index])
            print(len(np.where(condition == True)[0]))
            print(fun_index_f)
    elastic_i = np.sum(np.array(D_ij_list))+np.sum(np.array(D_iF_list))
    elastic_j = np.array(D_ij_list) + np.array(D_iF_list)
    return D_ij_list, D_iF_list, fun_index_array, fun_f_list,elastic_i, elastic_j

#梯度下降法
def Gradient_descent(H, K, xy_new, fun_index_array, fun_f_list):
    x=Symbol("x")
    y=Symbol("y")
    z=Symbol("z")
    
    x1=Symbol("x1")
    y1=Symbol("y1")
    z1=Symbol("z1")
    xdevice_gra_list = []
    ydevice_gra_list = []
    xenbeding_gra_list = []
    yenbeding_gra_list = []
    for i_num in range(len(xy_new)):
        x_gra_list = []
        y_gra_list = []
        for j_num in range(len(xy_new)):
            if i_num != j_num:
                if fun_index_array[i_num][j_num] == 0:
                    z = -y
                if fun_index_array[i_num][j_num] == 1:
                    z = -x
                if fun_index_array[i_num][j_num] == 2:
                    z = y
                if fun_index_array[i_num][j_num] == 3:
                    z = x
                if fun_index_array[i_num][j_num] == 4:
                    z = 0
            else:
                z = 0
            w0=xy_new[i_num][0:2]
            gx=diff(z,x).subs([(x,w0[0]),(y,w0[1])])#对x求偏导
            gy=diff(z,y).subs([(x,w0[0]),(y,w0[1])])#对y求偏导
            x_gra_list.append(gx)
            y_gra_list.append(gy)
        x_particle = float(np.sum(np.array(x_gra_list)))
        y_particle = float(np.sum(np.array(y_gra_list)))
        xdevice_gra_list.append(x_particle)
        ydevice_gra_list.append(y_particle)
        
        x_f = H/2
        y_f = K/2
        if fun_f_list[i_num] == 0:
            if xy_new[i_num][0] >= x_f:
                z1 = sqrt((x1-x_f+xy_new[i_num][2]/2-H/2)**2+xy_new[i_num][3]**2)
            if xy_new[i_num][0] < x_f:
                z1 = sqrt((x_f-x1+xy_new[i_num][2]/2-H/2)**2+xy_new[i_num][3]**2)
        if fun_f_list[i_num] == 1:
            if xy_new[i_num][1] >= y_f:
                z1 = sqrt((y1-y_f+xy_new[i_num][3]/2-K/2)**2+xy_new[i_num][2]**2)
            if xy_new[i_num][1] < y_f:
                z1 = sqrt((y_f-y1+xy_new[i_num][3]/2-K/2)**2+xy_new[i_num][2]**2)
        if fun_f_list[i_num] == 2:
            if xy_new[i_num][0] >= x_f and xy_new[i_num][1] >= y_f:
                z1 = sqrt((x1-x_f+xy_new[i_num][2]/2-H/2)**2+(y1-y_f+xy_new[i_num][3]/2-K/2)**2)
            if xy_new[i_num][0] >= x_f and xy_new[i_num][1] < y_f:
                z1 = sqrt((x1-x_f+xy_new[i_num][2]/2-H/2)**2+(y_f-y1+xy_new[i_num][3]/2-K/2)**2)
            if xy_new[i_num][0] < x_f and xy_new[i_num][1] >= y_f:
                z1 = sqrt((x_f-x1+xy_new[i_num][2]/2-H/2)**2+(y1-y_f+xy_new[i_num][3]/2-K/2)**2)
            if xy_new[i_num][0] < x_f and xy_new[i_num][1] < y_f:
                z1 = sqrt((x_f-x1+xy_new[i_num][2]/2-H/2)**2+(y_f-y1+xy_new[i_num][3]/2-K/2)**2)
        if fun_f_list[i_num] == 4:
            z1 = 0
        
        gx1=diff(z1,x1).subs([(x1,w0[0]),(y1,w0[1])])#对x1求偏导
        gy1=diff(z1,y1).subs([(x1,w0[0]),(y1,w0[1])])#对y1求偏导
        xenbeding_gra_list.append(float(gx1))
        yenbeding_gra_list.append(float(gy1))
    x_new = np.array(xdevice_gra_list)+np.array(xenbeding_gra_list)
    y_new = np.array(ydevice_gra_list)+np.array(yenbeding_gra_list)
#     print(y_new)
    p_xy_new = np.zeros((len(xy_new),4))
    p_xy_new[:,0:1] = xy_new[:,0:1]-x_new.reshape(len(xy_new),1)
    p_xy_new[:,1:2] = xy_new[:,1:2]-y_new.reshape(len(xy_new),1)
    p_xy_new[:,2:] = xy_new[:,2:]
    return p_xy_new

p_xy = pd.read_excel('设备位置.xlsx')
p_xy = pd.DataFrame(p_xy.values.T)
p_xy.columns = ['x','y','l','w']
prime = pd.read_excel('搬运成本.xlsx')
frequency = pd.read_excel('搬运数量.xlsx')
frequency = frequency.fillna(0)

#绘图
def random_color():
    color_list=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ''
    for i in range(6):
        color_number = color_list[random.randint(0,15)]
        color += color_number
    color = '#' +color
    return color
xy_num = 1
fig = plt.figure(figsize=(22.5, 10.5)) #创建图
ax = fig.add_subplot(111)
for index in range(13):
#     plt.scatter(e[index][0],e[index][1],color='black')
    x = p_xy.values[index][0]-p_xy.values[index][2]/2
    y = p_xy.values[index][1]-p_xy.values[index][3]/2
    color = random_color()
    rect = plt.Rectangle((x, y), p_xy.values[index][2], p_xy.values[index][3], fill=False, linewidth=3, edgecolor=color)
    ax.add_patch(rect)
    plt.text(p_xy.values[index][0],p_xy.values[index][1],index+1, fontsize=20)
plt.xlim(0, 225)
plt.ylim(0, 105)
plt.legend

#OSD计算粒子适应度
def OSD(MH_SR, ki, count):
    di = (np.amax(MH_SR, axis=0)-np.amin(MH_SR, axis=0))/ki
    ti = MH_SR-np.amin(MH_SR, axis=0)
    if (di != 0).all:
        hi = np.floor(ti/di)+1
    elif di[0] == 0 and di[0] != 0:
        hi = np.floor(ti/di)+1
        hi[:,0:1] = 0 
    elif di[0] != 0 and di[1] == 0:
        hi = np.floor(ti/di)+1
        hi[:,1:2] = 0 
    else:
        hi = ti*0
    MH_count = [np.count_nonzero(hi[:,0:1].flatten() != hi[i][0], axis=0) for i in range(len(MH_SR))]
    SR_count = [np.count_nonzero(hi[:,1:2].flatten() != hi[i][1], axis=0) for i in range(len(MH_SR))]
    num_count = np.array(MH_count)+np.array(SR_count)
    fit = np.array(count)/num_count
    return fit

#定义10个粒子参与计算
def PSO_algorithm(p_xy_array, Pn, itera, frequency, prime):
    H = 225
    K = 105
    H_oj = K_oj = 7.5
    d_jk = 3.5
    w = 0.8
    c1 = 1.2
    c2 = 1.2
    r1 = 0.6
    r2 = 0.6
    t1 = 500
    t2 = 100
    Dim = len(p_xy_array)
    
    XYposition = np.ones((Pn, Dim, 4))*p_xy_array             #10*13*4
    XYposition[1:5,:,0:2] = XYposition[1:5,:,0:2]*0.8
    Vspeed = np.zeros((Pn, Dim, 2))                           #10*13*2
    
    Pbest = XYposition                                        #10*13*4
    Gbest = p_xy_array                                        #13*4
    
    P_fit = np.zeros((Pn, ))                                  #(10,)
    G_fit = 0                                                 #这里的适应度是越大越好
    
    MH_SR = np.zeros((Pn, 2))
    MHSR_pbest = np.zeros((Pn, 2))                            #个体最优解适应值
    MHSR_pbest[:,0:1] = 10000
    Pareto = []                                               #pareto解外部集合
    Pareto_fit_list = []                                      #帕累托解对应适应值
    
    for itera_index in range(itera):
        #计算优化目标函数值
        for Pn_index in range(Pn):
            distance = np.array(dict_all(XYposition[Pn_index], Dim))
            MH = Objective_function_MH(distance, frequency, prime)
            SR = Objective_function_SR(XYposition[Pn_index], Dim)
            MH_SR[Pn_index][0] = MH
            MH_SR[Pn_index][1] = SR
        #寻找非支配解并统计支配数量
        pareto_count_list = []                     #储存粒子支配的个数，用于计算适应度
        Pareto_fit_index = []                      #帕累托粒子索引
        for Pn_i in range(Pn):
            count = 0
            if_pareto = True
            for Pn_j in range(Pn):
                if MH_SR[Pn_i][0]>MH_SR[Pn_j][0] and MH_SR[Pn_i][1]<MH_SR[Pn_j][1]:
                    if_pareto = False
                if (MH_SR[Pn_i][0]<=MH_SR[Pn_j][0]) and (MH_SR[Pn_i][1]>=MH_SR[Pn_j][1]):
                    count += 1

            if if_pareto:
                Pareto.append(XYposition[Pn_i])
                Pareto_fit_index.append(Pn_i)
                
            pareto_count_list.append(count)

        #OSD计算粒子适应度
        fit = OSD(MH_SR, 10, pareto_count_list)
        Pareto_fit_list = Pareto_fit_list + list(fit[Pareto_fit_index])
        XY_copy = XYposition
#         print(Pareto_fit_index)
#         print(Pareto_fit_list)
        
        #根据策略更新Pbest和Gbest
        for update_index in range(Pn):
            if MH_SR[update_index][0]< MHSR_pbest[update_index][0] and MH_SR[update_index][1] > MHSR_pbest[update_index][1]:
                Pbest[update_index] = XYposition[update_index]
                MHSR_pbest[update_index][0] = MH_SR[update_index][0]
                MHSR_pbest[update_index][1] = MH_SR[update_index][1]
            elif MH_SR[update_index][0] > MHSR_pbest[update_index][0] and MH_SR[update_index][1] < MHSR_pbest[update_index][1]:
                break
            else:
                if fit[update_index] > P_fit[update_index]:
                    Pbest[update_index] = XYposition[update_index]
                    MHSR_pbest[update_index][0] = MH_SR[update_index][0]
                    MHSR_pbest[update_index][1] = MH_SR[update_index][1]
                    
        Pareto_fit_array = np.array(Pareto_fit_list)
        pareto_sum = np.sum(Pareto_fit_array)
        Pareto_fit_div = Pareto_fit_array/pareto_sum
        Pareto_fit_div_sum = np.array([np.sum(Pareto_fit_div[0:i]) for i in range(len(Pareto_fit_list))])
        
#         print(Pareto_fit_div_sum)
        gbest_threshold = random.uniform(0.4, 0.75)
#         print(np.where(Pareto_fit_div_sum >= gbest_threshold))
        g_index = int(np.where(Pareto_fit_div_sum >= gbest_threshold)[0][0])
#         print(g_index)
        Gbest = Pareto[g_index]
        G_fit = Pareto_fit_list[g_index]
#         print(G_fit)
        
        Vspeed = w*Vspeed + c1*r1*(Pbest[:,:,0:2] - XYposition[:,:,0:2]) + c1*c2*(Gbest[:,0:2] - XYposition[:,:,0:2])
        XYposition[:,:,0:2] = XYposition[:,:,0:2] + Vspeed
        
        #粒子群更新完之后，开始变异，首先计算粒子挤压弹性势能
        elastic_all = []
        for Pn_count in range(len(XYposition)):
            elastic = Elastic_energy(H, K, H_oj, K_oj, XYposition[Pn_count])[4]
            elastic_all.append(elastic)
        elastic_all = np.array(elastic_all)    
        if (elastic_all>0).any():
            #存在粒子不合法,找出不合法粒子，找到最大挤压弹性势能设备，变异
            variation_index = list(np.where(elastic_all>0)[0])
#             print(variation_index)
            for var_index in variation_index:
                elastic = np.array(Elastic_energy(H, K, H_oj, K_oj, XYposition[var_index])[5])
                var_ = int(np.where(elastic == max(elastic))[0])
                randomxy = Random_xy(H, K, 1, H_oj, K_oj, XYposition[var_index:var_index+1,var_:var_+1:][0][0], XYposition[var_index])
                XYposition[var_index][var_][0:2] = randomxy
        else:
            #粒子合法找到最大物流量设备
            variation_index = list(np.where(elastic_all<=0)[0])
            for var_index in variation_index:
                distance_array = np.array(dict_all(XYposition[var_index], Dim))
                material = Material_charges(distance_array, frequency, prime)
                var_ = int(np.where(material == max(material))[0])
                randomxy = Random_xy(H, K, 1, H_oj, K_oj, XYposition[var_index:var_index+1,var_:var_+1:][0][0], XYposition[var_index])
                XYposition[var_index][var_][0:2] = randomxy
        #梯度下降合法化
        gra_num = 0
        while(gra_num<20):
            for gra_idnex in range(len(XYposition)):
                fun_index_array, fun_f_list = Elastic_energy(H, K, H_oj, K_oj, XYposition[gra_idnex])[2:4]
                XYposition[gra_idnex] = Gradient_descent(H, K, XYposition[gra_idnex], fun_index_array, fun_f_list)
            gra_num += 1
        #将新方案和种群融合
        MH_SR_new = np.zeros((Pn, 2))
        for Pn_index_new in range(len(XYposition)):
            distance_new = np.array(dict_all(XYposition[Pn_index_new], Dim))
            MH_new = Objective_function_MH(distance_new, frequency, prime)
            SR_new = Objective_function_SR(XYposition[Pn_index_new], Dim)
            MH_SR_new[Pn_index_new][0] = MH
            MH_SR_new[Pn_index_new][1] = SR
        XY_ALL = np.concatenate((XY_copy,XYposition))
        MH_SR_ALL = np.concatenate((MH_SR_new, MH_SR))
        #计算pareto更新EA
        updata_pareto_num = []                     #储存粒子支配的个数，用于计算适应度
        updata_pareto_index = []                      #帕累托粒子索引
        Pareto_new = []
        for Pn_i in range(len(XY_ALL)):
            count = 0
            if_pareto = True
            for Pn_j in range(Pn):
                if MH_SR_ALL[Pn_i][0]>MH_SR_ALL[Pn_j][0] and MH_SR_ALL[Pn_i][1]<MH_SR_ALL[Pn_j][1]:
                    if_pareto = False
                if (MH_SR_ALL[Pn_i][0]<=MH_SR_ALL[Pn_j][0]) and (MH_SR_ALL[Pn_i][1]>=MH_SR_ALL[Pn_j][1]):
                    count += 1

            if if_pareto:
                Pareto_new.append(XY_ALL[Pn_i])
                updata_pareto_index.append(Pn_i)
            
            updata_pareto_num.append(count)

        #OSD计算更新后粒子适应度
        fit_new = OSD(MH_SR_ALL, 10, updata_pareto_num)
        Pareto_fit_list = Pareto_fit_list + list(fit_new[updata_pareto_index])
        Pareto = np.concatenate((Pareto,Pareto_new))
        Pareto_save_index = sorted(range(len(Pareto_fit_list)), key=lambda k: Pareto_fit_list[k])[-20:]
        Pareto_fit_list = sorted(Pareto_fit_list)[-20:]
        Pareto = list(Pareto[Pareto_save_index])
        print(itera_index)
    return XYposition, Pareto, Pareto_fit_list
    
XYposition, Pareto, Pareto_fit_list = PSO_algorithm(p_xy.values, 10, 2, frequency.values, prime.values)

num_count = 0
E = 1000
xy_new = p_xy.values
e_list  = []
n = 0
best_copy = np.zeros((len(xy_new), 4))
while(num_count < 20):
    while(E>1 and n<100):
        device_e, enbeding_e, device_fun, enbeding_fun = Elastic_energy(225, 105, 7.5, 7.5, xy_new)
        xy_gra_gen = Gradient_descent(225, 105, xy_new, device_fun, enbeding_fun)
        E_di,E_eni,d_fi,e_fi = Elastic_energy(225, 105, 7.5, 7.5, xy_new)
        xy_new = xy_gra_gen
        E1 = sum(E_di)+sum(E_eni)
        if E1<E:
            best = xy_new
        E = E1
        e_list.append(E)
        n = n+1

    en_area = Embedded_around(225, 105, 7.5, 7.5, best)
    count = best
    for i in range(len(best)):
        rect_ = list(best[i])
        area_ = en_area[i]
        if area_ != 0:
            try:
                result, con = Boundary_transform(225, 105, 7.5, 7.5, rect_, area_)
                if len(np.where(con == True)[0]) == 1:
                    count[i] = result[0]
                    E_di,E_eni,d_fi,e_fi = Elastic_energy(225, 105, 7.5, 7.5, count)
                    e_tri1 = sum(E_di)+sum(E_eni)
                    count[i] = result[1]
                    E_di,E_eni,d_fi,e_fi = Elastic_energy(225, 105, 7.5, 7.5, count)
                    e_tri2 = sum(E_di)+sum(E_eni)
                    count[i] = result[2]
                    E_di,E_eni,d_fi,e_fi = Elastic_energy(225, 105, 7.5, 7.5, count)
                    e_tri3 = sum(E_di)+sum(E_eni)
                    etri = np.array([e_tri1,e_tri2,e_tri3])
                    ertri_index = np.where(etri==min(etri))[0]
                    if ertri_index == 0:
                        best_copy[i] = result[0]
                    if ertri_index == 1:
                        best_copy[i] = result[1]
                    if ertri_index == 2:
                        best_copy[i] = result[2]
                if len(np.where(con == True)[0]) == 2:
                    count[i] = result[0]
                    E_di,E_eni,d_fi,e_fi = Elastic_energy(225, 105, 7.5, 7.5, count)
                    e_tri1 = sum(E_di)+sum(E_eni)
                    count[i] = result[1]
                    E_di,E_eni,d_fi,e_fi = Elastic_energy(225, 105, 7.5, 7.5, count)
                    e_tri2 = sum(E_di)+sum(E_eni)
                    etri = np.array([e_tri1,e_tri2])
                    ertri_index = np.where(etri==min(etri))[0]
                    if ertri_index == 0:
                        best_copy[i] = result[0]
                    if ertri_index == 1:
                        best_copy[i] = result[1]
            except:
                break
#                 print(i)
        else:
            best_copy[i] = best[i]
        xy_new = best
    num_count = num_count+1
    
fig = plt.figure(figsize=(22.5, 10.5)) #创建图
ax = fig.add_subplot(111)
for index in range(13):
#     plt.scatter(e[index][0],e[index][1],color='black')
    x = Pareto[1][index][0]-Pareto[1][index][2]/2
    y = Pareto[1][index][1]-Pareto[1][index][3]/2
    color = random_color()
    rect = plt.Rectangle((x, y), Pareto[1][index][2], Pareto[1][index][3], fill=False, linewidth=3, edgecolor=color)
    ax.add_patch(rect)
    plt.text(Pareto[1][index][0],Pareto[1][index][1],index+1, fontsize=20)
plt.xlim(0, 225)
plt.ylim(0, 105)
plt.legend

#定义弹性势能计算公式
def Elastic_energy(H, K, H_oj, K_oj, xy_array):
    device_area = Embedding_device(xy_array)
    D_ij_list = []
    D_iF_list = []
    fun_index_array = np.ones((len(xy_array), len(xy_array)))*4
    for i in range(len(xy_array)):
        D_ij_step = []
        for j in range(len(xy_array)):
            if i != j:
                if device_area[i][j] > 0:
                    top_move = abs((xy_array[i][3]+xy_array[j][3])/2-(xy_array[i][1]-xy_array[j][1]))+3.5
                    right_move = abs((xy_array[i][2]+xy_array[j][2])/2-(xy_array[i][0]-xy_array[j][0]))+3.5
                    bottom_move = abs((xy_array[i][3]+xy_array[j][3])/2-(xy_array[j][1]-xy_array[i][1]))+3.5
                    left_move = abs((xy_array[i][2]+xy_array[j][2])/2-(xy_array[j][0]-xy_array[i][0]))+3.5
                    values = np.array([top_move, right_move, bottom_move, left_move])
                    D_ij = min(top_move, right_move, bottom_move, left_move)
                    fun_index = np.where(values == D_ij)[0][0]
                    fun_index_array[i][j] = fun_index
                else:
                    D_ij = 0
            else:
                D_ij = 0
            D_ij_step.append(D_ij)
        D_ij_list.append(float(np.sum(np.array(D_ij_step))))
        
    enbedding_area = Embedded_around(H, K, H_oj, K_oj, xy_array)
    x_f = H/2
    y_f = K/2
    fun_f_list = []
    for device_index in range(len(enbedding_area)):
        #这里不知道为什么，会产生一些很小很小的面积，导致报错。
        try:
            if (enbedding_area[device_index] < 0.5) and (enbedding_area[device_index] > 0):
                D_iF = 0
                fun_index_f = 4
                enbedding_area[device_index] == 0
            elif enbedding_area[device_index] >= 0.5:
                rect = xy_array[device_index]
                condition1 = rect[0]+rect[2]/2>=H-H_oj     #右边
                condition2 = rect[0]-rect[2]/2<=H_oj       #左边
                condition3 = rect[1]+rect[3]/2>=K-K_oj     #上边
                condition4 = rect[1]-rect[3]/2<=K_oj       #下边
                condition = np.array([condition1, condition2, condition3, condition4])
                con_ture = np.array([rect[0]+rect[2]/2<H_oj, rect[0]-rect[2]/2>H-H_oj, rect[1]+rect[3]/2<K_oj, rect[1]-rect[3]/2>K-K_oj]).any()
                if con_ture:
                    D_iF = 10000
                    fun_index_f = 2
                else:
                    if len(np.where(condition == True)[0]) == 1:
                        if condition[0] or condition[1]:
                            D_iF = np.sqrt((abs(xy_array[device_index][0]-x_f)+xy_array[device_index][2]/2-H/2+H_oj)**2+xy_array[device_index][3]**2)
                            fun_index_f = 0
                        elif condition[2] or condition[3]:
                            D_iF = np.sqrt((abs(xy_array[device_index][1]-y_f)+xy_array[device_index][3]/2-K/2+K_oj)**2+xy_array[device_index][2]**2)
                            fun_index_f = 1
                    elif len(np.where(condition == True)[0]) == 2:
                        D_iF = np.sqrt((abs(xy_array[device_index][1]-y_f)+xy_array[device_index][3]/2-K/2+K_oj)**2+\
                                    (abs(xy_array[device_index][0]-x_f)+xy_array[device_index][2]/2-H/2+H_oj)**2)
                        fun_index_f = 2
            else:
                D_iF = 0
                fun_index_f = 4
            D_iF_list.append(D_iF)
            fun_f_list.append(fun_index_f)
        except:
            print((rect),enbedding_area[device_index])
            print(len(np.where(condition == True)[0]))
            print(fun_index_f)
    elastic_i = np.sum(np.array(D_ij_list))+np.sum(np.array(D_iF_list))
    elastic_j = np.array(D_ij_list) + np.array(D_iF_list)
#     return D_ij_list, D_iF_list, fun_index_array, fun_f_list,elastic_i, elastic_j
    return elastic_i

import heapq
class Solution:
    def getSkyline(self, buildings):
        # 保存所有的点
        points = []
        for build in buildings:
            points.append((build[0], -build[2]))
            points.append((build[1],  build[2]))
        points.sort(key=lambda x: (x[0], x[1]))

        # 用来保存当前扫描线所经过的建筑的高度，用堆来表示
        heights = []
        res = []
        # 因为在堆里面删除元素需要更多的时间开销，所以先把需要删除的元素保存起来
        should_del = {} 

        # 保存当前的高度
        cur_height = 0

        for point in points:
            if point[1] < 0: heapq.heappush(heights, point[1])
            elif should_del.get(point[1]): 
                should_del[point[1]] += 1 # 保存需要删除的次数，删除的时候，删除一次
            else:
                should_del[point[1]] = 1

            # 如果当前堆顶元素是应该删除的元素就先删除掉
            while heights and -heights[0] in should_del:
                temp = -heights[0]
                heapq.heappop(heights)
                should_del[temp] -= 1
                if should_del[temp] == 0:
                    should_del.pop(temp)
            
            maxH = -heights[0] if heights else 0
            if maxH != cur_height:
                cur_height = maxH
                res.append([point[0], cur_height])

        return res
