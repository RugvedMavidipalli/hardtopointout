from mp_run import run_testbed
from multiprocessing import Process, Manager
should_train = False
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import plotly.graph_objects as go
from matplotlib import cm


manager = Manager()
testbed_data1 = manager.dict()

testbed_data1['train'] = False
testbed_data1['render'] = False
testbed_data1['load_new_data'] = False
testbed_data1['render_depth'] = False

class Config:
    def __init__(self):
        self.output_path = 'output'
        self.images_path = 'data/demo'

config = Config()

# p1 = Process(target=run_testbed, args=['../../configs/nerf', f'{config.images_path}/transforms.json', f'{config.output_path}', testbed_data1, 0])
# p1.start()
# testbed_data1['train'] = True
# time.sleep(60)
#
# testbed_data1['render_depth'] = True

while testbed_data1['render_depth']:
    pass

def load_transforms(transforms_path):
    with open(transforms_path, 'r') as f:
        json_data = json.load(f)
    extrinsics = { int(frame['file_path'][frame['file_path'].rfind('/') + 1:-4]): np.array(frame['transform_matrix']) for frame in json_data['frames'] }
    intrinsics = np.array([[json_data['fl_x'], 0, json_data['cx']], [0, json_data['fl_y'], json_data['cy']], [0, 0, 1]])
    return intrinsics, extrinsics

def load_depths(root, paths):
    depths = {}
    for path in paths:
        full_path = f'{root}/{path}_depth.npy'
        if os.path.exists(full_path):
            depths[path] = np.load(full_path)
            depths[path][depths[path] < .1] = 0
    return depths

def load_colors(root, paths):
    colors = {}
    for path in paths:
        full_path = f'{root}/{path}.jpg'
        if os.path.exists(full_path):
            colors[path] = np.array(Image.open(full_path))
    return colors
def backproject(intrinsics, extrinsics, depth, color):
    intrinsics_inverse = np.linalg.inv(intrinsics)
    out = []
    colors = []
    quasinormals = []
    center = extrinsics[:3, 3]
    for px in range(0, depth.shape[1], 10):
        for py in range(0, depth.shape[0], 10):
            camera_xyz = (intrinsics_inverse @ np.array([px, py, 1])) * (1 / np.exp(0.6931471805599453 * -5)) * -depth[py, px, 0]
            world_xyz = (extrinsics @ np.array([-camera_xyz[0], camera_xyz[1], camera_xyz[2], 1]))[:3]
            out.append(world_xyz)
            colors.append(color[py, px, :3])
            quasinormals.append(world_xyz - center)
    # x_linspace = np.linspace(0, depth.shape[1] - 1, int(depth.shape[1] / 20))
    # y_linspace = np.linspace(0, depth.shape[0] - 1, int(depth.shape[0] / 20))
    # pxs, pys = np.meshgrid(x_linspace, y_linspace)
    # pxs = pxs.reshape(-1)
    # pys = pys.reshape(-1)
    # ds = np.take(depth.reshape(-1), (pys * depth.shape[0] + pxs).astype(np.int32))
    # homogenous = np.ones((3, pxs.shape[0]))
    # homogenous[0, :] = pxs
    # homogenous[1, :] = pys
    # camera_xyz = (intrinsics_inverse @ homogenous) * (1 / np.exp(0.6931471805599453 * -5)) * -ds
    # world_xyz = (extrinsics @ np.concatenate([camera_xyz, np.ones((1, camera_xyz.shape[1]))], 0))[:3]
    # quasinormals = world_xyz - extrinsics[:3, 3:]
    # return np.concatenate([world_xyz, quasinormals], 0)
    return out, colors, quasinormals

def project(intrinsics, h, w, input_h, input_w, extrinsics, points, quasinormals, colors):
    extrinsics_inverse = np.linalg.inv(extrinsics)
    out_depth = np.ones((h, w)) * -10
    out_color = np.zeros((h, w, 3))
    out_qn = np.zeros((h, w))
    for p, qn, color in zip(points, quasinormals, colors):
        camera_xyz = extrinsics_inverse @ np.array([p[0], p[1], p[2], 1])
        camera_camera_center = extrinsics_inverse @ np.array([p[0] + qn[0], p[1] + qn[1], p[2] + qn[2], 1])
        camera_xyz[0] = -camera_xyz[0]
        image_xyz = intrinsics @ camera_xyz[:3]
        if image_xyz[2] == 0:
            continue
        px = int(image_xyz[0] / image_xyz[2] * w / input_w)
        py = int(image_xyz[1] / image_xyz[2] * h / input_h)
        if py < h and px < w and py >= 0 and px >= 0 and camera_xyz[2] > out_depth[py, px] and camera_xyz[2] < -0.1:
            out_depth[py, px] = camera_xyz[2]
            out_color[py, px] = color
            out_qn[py, px] = 1 if camera_xyz[2] > camera_camera_center[2] else 0
    return out_depth, out_color, out_qn

intrinsics, input_transforms = load_transforms(config.images_path+ '/transforms.json')
depths = load_depths(config.output_path, input_transforms.keys())
colors = load_colors(config.images_path, input_transforms.keys())
print(len(depths))
print(len(colors))
# for i in depths:
#     fig, axs = plt.subplots(1, 2)
#     axs[0].imshow(depths[i])
#     axs[1].imshow(colors[i])
#     plt.show()
xyz = None
quasinormals = None
count = 0
color = []
keys = [x for x in input_transforms.keys()]
for i in keys:
    if i in depths:
        count += 1
        new_xyz, new_color, new_quasinormals = backproject(intrinsics, input_transforms[i], depths[i], colors[i])
        if xyz is None:
            xyz = new_xyz
            quasinormals = new_quasinormals
        else:
            xyz = np.concatenate([xyz, new_xyz], 0)
            quasinormals = np.concatenate([quasinormals, new_quasinormals], 0)
        color += new_color
locs = np.zeros((len(keys), 3))
loc_color = [0]*len(keys)
for key in keys:
    locs[key, :] = input_transforms[key][:3, 3]
    rendered_depth, rendered_color, rendered_qn = project(intrinsics, 120, 216, 720, 1280, input_transforms[key], xyz, quasinormals, color)
    mask = rendered_depth != -10
    all = np.sum(mask)
    good = np.sum(rendered_qn[mask])
    print(good / all)
    if key in depths:
        loc_color[key] = 2
    else:
        loc_color[key] = good / all

viridis = cm.get_cmap('viridis', 12)
new_loc_color = [x for x in viridis(np.array(loc_color))]
new_loc_color = [x if y != 2 else (1, 0, 0, 1) for x, y in zip(new_loc_color, loc_color)]
color += new_loc_color

xyz = np.concatenate([xyz, locs], 0)

fig = go.Figure(data=[go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode='markers',
        marker=dict(
            size=8,
            color=color,
            opacity=1
        )
    )])
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig, axs = plt.subplots(1, 3)
axs[0].imshow(rendered_color / 255)
axs[1].imshow(rendered_depth)
axs[2].imshow(rendered_qn)
plt.show()
print(len(color))