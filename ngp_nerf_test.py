from mp_run import run_testbed
from multiprocessing import Process, Manager
should_train = False
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# manager = Manager()
# d = manager.dict()
#
# d['train'] = False
# d['load_new_data'] = False
# d['completed'] = False
# d['render'] = False
#
# p = Process(target=run_testbed, args=['../../configs/nerf', 'data/demo/transforms.json', 'output', d])
# p.start()
# while True:
#     cmd = input()
#     if cmd == 's':
#         d['train'] = True
#     if cmd == 'l':
#         d['load_new_data'] = True
#     if cmd == 'e':
#         d['train'] = False
#     if cmd == 'c':
#         d['train'] = False
#         d['completed'] = True
#     if cmd == 'r':
#         d['render'] = True

manager = Manager()
testbed_data1 = manager.dict()
testbed_data2 = manager.dict()

testbed_data1['train'] = False
testbed_data2['train'] = False
testbed_data1['render'] = False
testbed_data2['render'] = False
testbed_data1['load_new_data'] = False
testbed_data2['load_new_data'] = False

class Config:
    def __init__(self):
        self.output_path = 'output'
        self.images_path = 'data/demo'

config = Config()

p1 = Process(target=run_testbed, args=['../../configs/nerf', f'{config.images_path}/transforms.json', f'{config.output_path}', testbed_data1, 0])
p1.start()
p2 = Process(target=run_testbed, args=['../../configs/nerf', f'{config.images_path}/transforms.json', f'{config.output_path}', testbed_data2, 1])
p2.start()
testbed_data1['train'] = True
testbed_data2['train'] = True
time.sleep(10)



testbed_data1['render'] = True
testbed_data2['render'] = True

with open(f'{config.images_path}/transforms.json', 'r') as f:
    transforms = json.load(f)
vals = np.zeros(len(transforms['frames']))
locs = np.zeros((len(transforms['frames']), 3))
while testbed_data1['render'] or testbed_data2['render']:
    pass
fig, ax = plt.subplots(16, 6, squeeze=True)
for frame in transforms['frames']:
    index = int(frame['file_path'][frame['file_path'].rfind('/') + 1:-4])
    locs[index, :] = np.array(frame['transform_matrix'])[:3, 3]
    image1 = np.load(f"{config.output_path}/{frame['file_path'][:-4]}_0.npy")
    image2 = np.load(f"{config.output_path}/{frame['file_path'][:-4]}_1.npy")
    ax[int(index / 6)][index % 6].imshow(np.sum(image1 - image2, 2) ** 2)
    vals[index] = np.mean((image1 - image2) ** 2)
plt.show()
def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.
    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)
def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])
def plot_point_mat(ax, mat, color, marker):
    ax.scatter(mat[:, 0], mat[:, 1], mat[:, 2], c=color, marker=marker, s=8)

vals -= np.min(vals)
vals /= np.max(vals)

viridis = cm.get_cmap('viridis', 12)
color = viridis(vals)
color[0] = (1, 0, 0, 1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plot_point_mat(ax, locs, color, 'o')
ax.set_box_aspect([1,1,1]) # IMPORTANT - this is the new, key line
# ax.set_proj_type('ortho') # OPTIONAL - default is perspective (shown in image above)
set_axes_equal(ax) # IMPORTANT - this is also required
plt.show()