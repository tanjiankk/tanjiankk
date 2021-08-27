- 👋 Hi, I’m @tanjiankk
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...

<!---
tanjiankk/tanjiankk is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
ret1_x1 = ret1[0]-ret1[2]/2
ret1_x2 = ret1[0]+ret1[2]/2
ret2_x1 = ret2[0]-ret2[2]/2
ret2_x2 = ret2[0]+ret2[2]/2

ret1_y1 = ret1[1]-ret1[3]/2
ret1_y2 = ret1[1]+ret1[3]/2
ret2_y1 = ret2[1]-ret2[3]/2
ret2_y2 = ret2[1]+ret2[3]/2

a = np.sort(np.array([ret1_x1, ret1_x2, ret2_x1, ret2_x2])).tolist()
b = np.sort(np.array([ret1_y1, ret1_y2, ret2_y1, ret2_y2])).tolist()

ox1 = [a[1], b[2]]
ox2 = [a[2], b[1]]

#计算面积

s1 = (max(ox1[0], ox2[0]) - min(ox1[0], ox2[0]))*ret1[3]
s2 = (max(ox1[1], ox2[1]) - min(ox1[1], ox2[1]))*ret1[2]
      
s3 = (max(ox1[0], ox2[0]) - min(ox1[0], ox2[0]))*ret2[3]
s4 = (max(ox1[1], ox2[1]) - min(ox1[1], ox2[1]))*ret2[2]
s_array = np.array([s1, s2, s3, s4])
index = np.where(s_array == min(s_array))[0][1]
if index == 0:
    if ret1[0]-ret1[2]/2 == ox1[0] or ret1[0]-ret1[2]/2 == ox2[0]:
        remove_xy = [ox1[0], ret1[1]+ret1[3]/2, ox2[0], ret1[1]-ret1[3]/2]
        wnew1 = ret1[3]
        lnew1 = ret1[2]
        xnew1 = remove_xy[2]+lnew1/2
        ynew1 = ret1[1]
        
        lnew2 = lnew3 = ret1[2]-min(s_array)/ret1[3]
        wnew2 = wnew3 = ret1[3]+min(s_array)/lnew2
        
        xnew2 = xnew3 = remove_xy[2]+lnew2/2
        ynew2 = remove_xy[3]+wnew2/2
        ynew3 = remove_xy[1]-wnew2/2
        
    if ret1[0]+ret1[2]/2 == ox1[0] or ret1[0]+ret1[2]/2 == ox2[0]:
        wnew1 = ret1[3]
        lnew1 = ret1[2]
        xnew1 = remove_xy[0]-lnew1/2
        ynew1 = ret1[1]
        
        lnew2 = lnew3 = ret1[2]-min(s_array)/ret1[3]
        wnew2 = wnew3 = ret1[3]+min(s_array)/lnew2
        
        xnew2 = xnew3 = remove_xy[0]-lnew2/2
        ynew2 = remove_xy[3]+wnew2/2
        ynew3 = remove_xy[1]-wnew2/2
if index == 1:
    if ret1[0]-ret1[2]/2 == ox1[0] or ret1[0]-ret1[2]/2 == ox2[0]:
        remove_xy = [ox1[0], ox1[1], ret1[0]+ret1[2]/2, ox2[1]]
        wnew1 = ret1[3]
        lnew1 = ret1[2]
        xnew1 = ret1[0]
        ynew1 = remove_xy[1]+wnew1/2
        
        wnew2 = wnew3 = ret1[3]-min(s_array)/ret1[2]
        lnew2 = lnew3 = ret1[2]+min(s_array)/wnew2
        
        ynew2 = ynew3 = remove_xy[1]+wnew2/2
        xnew2 = remove_xy[2]-lnew2/2
        xnew3 = remove_xy[0]+lnew2/2
    if ret1[0]+ret1[2]/2 == ox1[0] or ret1[0]+ret1[2]/2 == ox2[0]:
        remove_xy = [ret1[0]+ret1[2]/2, ox1[1], ox2[0], ox2[1]]
        wnew1 = ret1[3]
        lnew1 = ret1[2]
        xnew1 = ret1[0]
        ynew1 = remove_xy[1]+wnew1/2
        
        wnew2 = wnew3 = ret1[3]-min(s_array)/ret1[2]
        lnew2 = lnew3 = ret1[2]+min(s_array)/wnew2
        
        ynew2 = ynew3 = remove_xy[1]+wnew2/2
        xnew2 = remove_xy[2]-lnew2/2
        xnew3 = remove_xy[0]+lnew2/2
if index == 2:
    if ret2[0]-ret2[2]/2 == ox1[0] or ret2[0]-ret2[2]/2 == ox2[0]:
        remove_xy = [ox1[0], ox1[1], ox2[0], ret2[1]-ret2[3]/2]
        wnew1 = ret2[3]
        lnew1 = ret2[2]
        xnew1 = remove_xy[2]+lnew1/2
        ynew1 = ret2[1]
        
        lnew2 = lnew3 = ret2[2]-min(s_array)/ret2[3]
        wnew2 = wnew3 = ret2[3]+min(s_array)/lnew2
        
        xnew2 = xnew3 = remove_xy[2]+lnew2/2
        ynew2 = remove_xy[3]+wnew2/2
        ynew3 = remove_xy[1]-wnew2/2
    if ret2[0]+ret2[2]/2 == ox1[0] or ret2[0]+ret2[2]/2 == ox2[0]:
        remove_xy = [ox1[0], ox1[1], ox2[0], ret2[1]-ret2[3]/2]
        wnew1 = ret2[3]
        lnew1 = ret2[2]
        xnew1 = remove_xy[0]-lnew1/2
        ynew1 = ret2[1]
        
        lnew2 = lnew3 = ret2[2]-min(s_array)/ret2[3]
        wnew2 = wnew3 = ret2[3]+min(s_array)/lnew2
        
        xnew2 = xnew3 = remove_xy[0]-lnew2/2
        ynew2 = remove_xy[3]+wnew2/2
        ynew3 = remove_xy[1]-wnew2/2
if index == 3:
    if ret2[0]-ret2[2]/2 == ox1[0] or ret2[0]-ret2[2]/2 == ox2[0]:
        remove_xy = [ox1[0], ox1[1], ret2[0]+ret2[2]/2, ox2[1]]
        wnew1 = ret2[3]
        lnew1 = ret2[2]
        xnew1 = ret2[0]
        ynew1 = remove_xy[3]-wnew1/2
        
        wnew2 = wnew3 = ret2[3]-min(s_array)/ret2[2]
        lnew2 =lnew3 = ret2[2]+min(s_array)/wnew2
        
        ynew2 = ynew3 = remove_xy[3]-wnew2/2
        xnew2 = remove_xy[2]-lnew2/2
        xnew3 = remove_xy[0]+lnew2/2
        bb = 0
    if ret2[0]+ret2[2]/2 == ox1[0] or ret2[0]+ret2[2]/2 == ox2[0]:
        remove_xy = [ret2[0]-ret2[2]/2, ox1[1], ox2[0], ox2[1]]
        wnew1 = ret2[3]
        lnew1 = ret2[2]
        xnew1 = ret2[0]
        ynew1 = remove_xy[3]-wnew1/2
        
        wnew2 = wnew3 = ret2[3]-min(s_array)/ret2[2]
        lnew2 =lnew3 = ret2[2]+min(s_array)/wnew2
        
        ynew2 = ynew3 = remove_xy[3]-wnew2/2
        xnew2 = remove_xy[2]-lnew2/2
        xnew3 = remove_xy[0]+lnew2/2
        bb = 1
v1 = [xnew1,ynew1,lnew1,wnew1]
v2 = [xnew2,ynew2,lnew2,wnew2]
v3 = [xnew3,ynew3,lnew3,wnew3]

fig = plt.figure(figsize=(22.5, 10.5)) #创建图
ax = fig.add_subplot(111)

rect = plt.Rectangle((ret1[0]-ret1[2]/2, ret1[1]-ret1[3]/2), ret1[2], ret1[3], fill=False, linewidth=3, edgecolor='red')
ax.add_patch(rect)
rect = plt.Rectangle((ret2[0]-ret2[2]/2, ret2[1]-ret2[3]/2), ret2[2], ret2[3], fill=False, linewidth=3, edgecolor='red')
ax.add_patch(rect)

rect = plt.Rectangle((v1[0]-v1[2]/2, v1[1]-v1[3]/2), v1[2], v1[3], fill=False, linewidth=3, edgecolor='b')
ax.add_patch(rect)
rect = plt.Rectangle((v2[0]-v2[2]/2, v2[1]-v2[3]/2), v2[2], v2[3], fill=False, linewidth=3, edgecolor='k')
ax.add_patch(rect)
rect = plt.Rectangle((v3[0]-v3[2]/2, v3[1]-v3[3]/2), v3[2], v3[3], fill=False, linewidth=3, edgecolor='y')
ax.add_patch(rect)

plt.xlim(0, 20)
plt.ylim(0, 20)
plt.legend
plt.show()

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
    
    #计算设备与设备嵌入面积
def Embedding_device(xylw_new):
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
    
 embed_area_index = pd.Series(embed_area_list).sort_values(ascending=False).index.tolist()

    for index_ in range(len(xylw_new)):
        area_s = embed_area_list[embed_area_index[index_]]
        if area_s != 0:
            right = xylw_new[embed_area_index[index_]][0] + xylw_new[embed_area_index[index_]][2]/2 > H - H_oj
            left = xylw_new[embed_area_index[index_]][0] - xylw_new[embed_area_index[index_]][2]/2 < H_oj
            top = xylw_new[embed_area_index[index_]][1] + xylw_new[embed_area_index[index_]][3]/2 > K - K_oj
            bottom = xylw_new[embed_area_index[index_]][1] - xylw_new[embed_area_index[index_]][3]/2 < K_oj
            val_list = [right, left, top, bottom]
            true_num = [i for i, x in enumerate(val_list) if x]
            if len(true_num) = 1:
                x_new_list = []
                y_new_list = []
                if true_num[0]:
                    y_new_1 = xylw_new[embed_area_index[index_]][1]
                    x_new_1 = H-H_oj-xylw_new[embed_area_index[index_]][2]/2
                    y_new_2 = xylw_new[embed_area_index[index_]][1] + area_s/2*(H-H_oj-xylw_new[embed_area_index[index_]][0]-\
                              xylw_new[embed_area_index[index_]][2]/2)
                    x_new_1 = (H-H_oj+xylw_new[embed_area_index[index_]][0])/2 - xylw_new[embed_area_index[index_]][2]/4
                    y_new_3 = xylw_new[embed_area_index[index_]][1] - area_s/2*(H-H_oj-xylw_new[embed_area_index[index_]][0]-\
                              xylw_new[embed_area_index[index_]][2]/2)
                    x_new_3 = (H-H_oj+xylw_new[embed_area_index[index_]][0])/2 - xylw_new[embed_area_index[index_]][2]/4
                if true_num[1]:
                    
                if true_num[2]:
                    
                if true_num[3]:
                    
            if len(true_num) = 2:
            
            
#绘图
fig = plt.figure(figsize=(22.5, 10.5)) #创建图
ax = fig.add_subplot(111)
for index in range(13):
#     plt.scatter(e[index][0],e[index][1],color='black')
    x = e.values[index][0]-e.values[index][2]/2
    y = e.values[index][1]-e.values[index][3]/2
    color = random_color()
    rect = plt.Rectangle((x, y), e.values[index][2], e.values[index][3], fill=False, linewidth=3, edgecolor=color)
    ax.add_patch(rect)
plt.xlim(0, 225)
plt.ylim(0, 105)
plt.legend

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
for index in range(20):
    plt.scatter(random_xy[index][0],random_xy[index][1],color='black')
    x = random_xy[index][0]-p_xy.values[xy_num][2]/2
    y = random_xy[index][1]-p_xy.values[xy_num][3]/2
    color = random_color()
    rect = plt.Rectangle((x, y), p_xy.values[xy_num][2], p_xy.values[xy_num][3], fill=False, linewidth=3, edgecolor=color)
    ax.add_patch(rect)
plt.xlim(0, 225)
plt.ylim(0, 105)
plt.legend
plt.show()

xy_num = 1
fig = plt.figure(figsize=(22.5, 10.5)) #创建图
ax = fig.add_subplot(111)
for index in range(20):
    plt.scatter(random_xy[index][0],random_xy[index][1],color='black')
    x = random_xy[index][0]-p_xy.values[xy_num][3]/2
    y = random_xy[index][1]-p_xy.values[xy_num][2]/2
    color = random_color()
    rect = plt.Rectangle((x, y), p_xy.values[xy_num][3], p_xy.values[xy_num][2], fill=False, linewidth=3, edgecolor=color)
    ax.add_patch(rect)
plt.xlim(0, 225)
plt.ylim(0, 105)
plt.legend
plt.show()

fig = plt.figure(figsize=(22.5, 10.5)) #创建图
ax = fig.add_subplot(111)
for index in range(20):
    plt.scatter(random_xy[index][0],random_xy[index][1],color='black')
    try:
        x = p_xy.values[index][0]-p_xy.values[index][2]/2
        y = p_xy.values[index][1]-p_xy.values[index][3]/2
        color = random_color()
        rect = plt.Rectangle((x, y), p_xy.values[index][2], p_xy.values[index][3], facecolor=color)
        ax.add_patch(rect)
    except:
        print('.',end=' ')
plt.xlim(0, 225)
plt.ylim(0, 105)
plt.legend
plt.show()
