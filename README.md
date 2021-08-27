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
