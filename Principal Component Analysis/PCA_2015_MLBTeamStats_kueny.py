import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import pandas as pnd



class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
#file used for PCA is "data.csv" (attached) taken from population data
#2015 team stats

raw_data = pnd.read_csv('data.csv').as_matrix()


data = np.array(raw_data)

#get mean vector
mean_vec = data.mean(0)
meanx = mean_vec[0]
meany = mean_vec[1]
meanz = mean_vec[2]
#center the data by subtracting mean vector
for x in range(len(data)):
    for y in range(len(data[x])):
        data[x][y] -= mean_vec[y]

#covariance matrix and total_variance
cov_mat = np.cov(data.T)
total_variance = np.trace(cov_mat)

#get eig_vals and vecs and check variance
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
percents = [eig_vals[i]/total_variance for i in range(len(eig_vals))]

#make sure the eigen_vectors are ok
for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
#get eigen_pairs
eigen_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

#sort eigen pairs descending
eigen_pairs.sort()
eigen_pairs.reverse()

#variance shows that 4 components get us to 90% (At bats, hits, runs, singles)
eig_val_percents = [eigen_pairs[i][0]/total_variance for i in range(len(eigen_pairs))]

#remove other non-principal components
Q = np.vstack((eigen_pairs[0][1], eigen_pairs[1][1], eigen_pairs[2][1], eigen_pairs[3][1]))

new_data = Q.dot(data.T)
#use only 3 so we can plot
new_data = new_data[:3]
#new mean vector to center eigen_vectors
mean_vec = new_data.mean(1)
meanx = mean_vec[0]
meany = mean_vec[1]
meanz = mean_vec[2]

plt.xlim(min(new_data[0]) - 20, max(new_data[0]) + 20)
plt.ylim(min(new_data[1]) - 20, max(new_data[1]) + 20)
plt.scatter(new_data[0], new_data[1])
plt.ylabel('Runs')
plt.xlabel('At Bats')
plt.title('At Bats vs. Runs PCA')
plt.show()

# def init(new_data, mean_vec, data, eig_vecs, eig_vals):
fig2 = plt.figure(figsize=(8,8))
ax2 = fig2.add_subplot(111,projection='3d')
ax2.set_zlim([-3,3])
positive_vals =[]
not_pos_vals = []
test = new_data.T
print test.shape
pos = True
for x in range(len(test)):
    pos = True
    for y in range(len(test[x])):
        if test[x][y] < 0:
            pos = False
            not_pos_vals.append(test[x].tolist())
            continue
    if pos:
        positive_vals.append(test[x].tolist())
positive_vals = np.array(positive_vals).T
not_pos_vals = np.array(not_pos_vals).T
print positive_vals.shape
print not_pos_vals.shape


# new_data = new_data[:3]
# for x in range(len(new_data)):
#     for y in range(len(new_data))

ax2.plot(positive_vals[0], positive_vals[1], positive_vals[2], '^', markersize=10, color='green', alpha=0.5, label='At Bats, Hits, and Runs all above 0')
ax2.plot(not_pos_vals[0], not_pos_vals[1], not_pos_vals[2], 'o', markersize=10, color='red', alpha=0.5, label='At least one component value below 0')
ax2.plot([meanx], [meany], [meanz], 'o', markersize=20, color='blue', alpha=0.2)

print eig_vecs[0][0], eig_vecs[0][1], eig_vecs[0][2]
eig_vecs = eig_vecs
print eig_vecs
eig_vecs = eig_vecs[:3]
print eig_vecs
eig_vals = eig_vals[:3]
print eig_vals
colors = ['blue', 'yellow', 'red']
ctr = 0
axes_ = ['x ', 'y', 'z']
for v in eig_vecs:
    a = Arrow3D([meanx, -200.0*v[0]], [meany, 200.0*v[1]], [meanz, -200.0*v[2]], mutation_scale=15, lw=5, arrowstyle="-|>", color=colors[ctr], label=axes_[ctr] + 'component axis')
    a1 = Arrow3D([meanx, 200.0*v[0]], [meany, -200.0*v[1]], [meanz, 200.0*v[2]], mutation_scale=15, lw=5, arrowstyle="-|>", color=colors[ctr])
    ax2.add_artist(a)
    ax2.add_artist(a1)
    ctr += 1
plt.title('3D PCA of 2015 MLB Team Stats')
ax2.set_xlabel('x values (At Bats)')
ax2.set_ylabel('y_values (Hits)')
ax2.set_zlabel('z_values (Runs)')
plt.legend(bbox_to_anchor=(1, 0), loc=0, borderaxespad=0.)
ax2.set_xlim([-300,300])
ax2.set_ylim([-300,300])
ax2.set_zlim([-300,300])
# fig = plt.figure()
# ax = Axes3D(fig)
# def animate(i):
#     ax.view_init(elev=10.,azim=i)
# Animate
# anim = animation.FuncAnimation(fig, animate(i), init_func=init(new_data, mean_vec, data, eig_vecs, eig_vals),frames=360, interval=20, blit=True)
# # Save
# anim.save('kueny_pca3d_animation.mp4')
plt.show()
