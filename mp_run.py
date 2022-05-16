#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import commentjson as json

import numpy as np

import sys
import time

from common import *
from scenes import scenes_nerf, scenes_image, scenes_sdf, scenes_volume, setup_colored_sdf

from tqdm import tqdm

import pyngp as ngp  # noqa
import time
import threading


def run_testbed(config_dir, dataset_dir, output_dir, shared_data, index=0, gui=True, load_snapshot=None, snapshot_path=None):
    mode = ngp.TestbedMode.Nerf
    configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")
    scenes = scenes_nerf

    with open(dataset_dir, 'r') as f:
        test_transforms = json.load(f)

    base_network = os.path.join(configs_dir, "base.json")
    if dataset_dir in scenes:
        network = scenes[dataset_dir]["network"] if "network" in scenes[dataset_dir] else "base"
        base_network = os.path.join(configs_dir, network + ".json")
    network = base_network
    if not os.path.isabs(network):
        network = os.path.join(configs_dir, network)

    testbed = ngp.Testbed(mode)
    testbed.nerf.sharpen = float(0)

    scene = dataset_dir
    if not os.path.exists(dataset_dir) and dataset_dir in scenes:
        scene = os.path.join(scenes[dataset_dir]["data_dir"], scenes[dataset_dir]["dataset"])
    testbed.load_training_data(scene)

    if load_snapshot:
        print("Loading snapshot ", load_snapshot)
        testbed.load_snapshot(load_snapshot)
    else:
        testbed.reload_network_from_file(network)

    if gui:
        # Pick a sensible GUI resolution depending on arguments.
        sw = 1920
        sh = 1080
        while sw * sh > 1920 * 1080 * 4:
            sw = int(sw / 2)
            sh = int(sh / 2)
        testbed.init_window(sw, sh)

    testbed.shall_train = True

    testbed.nerf.render_with_camera_distortion = True

    network_stem = os.path.splitext(os.path.basename(network))[0]

    testbed.render_aabb = ngp.BoundingBox([0.049, 0.253, -0.130], [0.946, 0.763, 0.989])
    while testbed.frame():
        testbed.render_aabb = ngp.BoundingBox([0.049, 0.253, -0.130], [0.946, 0.763, 0.989])
        if testbed.want_repl():
            repl(testbed)
        testbed.shall_train = shared_data['train']
        if shared_data['load_new_data']:
            if gui:
                testbed.destroy_window()
            testbed.save_snapshot(f'{output_dir}/snapshot{index}.msgpack', False)
            testbed = ngp.Testbed(mode)
            testbed.load_snapshot(f'{output_dir}/snapshot{index}.msgpack')
            testbed.nerf.render_with_camera_distortion = True
            testbed.nerf.training.near_distance = 1
            testbed.shall_train = False

            testbed.load_training_data(dataset_dir)
            shared_data['load_new_data'] = False
            shared_data['train'] = True

            if gui:
                sw = 1920
                sh = 1080
                while sw * sh > 1920 * 1080 * 4:
                    sw = int(sw / 2)
                    sh = int(sh / 2)
                testbed.init_window(sw, sh)
            testbed.shall_train = True
        if shared_data['render']:
            for i, frame in enumerate(test_transforms["frames"]):
                testbed.render_aabb = ngp.BoundingBox([0.049, 0.253, -0.130], [0.946, 0.763, 0.989])
                testbed.set_nerf_camera_matrix(np.matrix(frame["transform_matrix"])[:-1, :])
                image = testbed.render(320, 180, 4, True)

                np.save(f"{output_dir}/{frame['file_path'][:-4]}_{index}.npy", image)
            shared_data['render'] = False
        if shared_data['render_depth']:
            for i, frame in enumerate(test_transforms["frames"]):
                print(f"{dataset_dir[:dataset_dir.rfind('/')]}/{frame['file_path']}")
                if not os.path.exists(f"{dataset_dir[:dataset_dir.rfind('/')]}/{frame['file_path']}"):
                    continue
                testbed.render_mode = ngp.RenderMode.Depth
                testbed.exposure = -5.0
                testbed.background_color = [0.0, 0.0, 0.0, 1.0]
                testbed.snap_to_pixel_centers = True
                testbed.fov_axis = 0
                testbed.fov = test_transforms["camera_angle_x"] * 180 / np.pi
                testbed.render_aabb = ngp.BoundingBox([0.049, 0.253, -0.130], [0.946, 0.763, 0.989])
                testbed.set_nerf_camera_matrix(np.matrix(frame["transform_matrix"])[:-1, :])
                image = testbed.render(1280, 720, 8, True)


                np.save(f"{output_dir}/{frame['file_path'][:-4]}_depth.npy", image)
            testbed.render_mode = ngp.RenderMode.Shade
            testbed.exposure = 0
            shared_data['render_depth'] = False