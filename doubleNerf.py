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

import pyngp as ngp # noqa
import gc

def parse_args():
    parser = argparse.ArgumentParser(description="Run neural graphics primitives testbed with additional configuration & output options")

    parser.add_argument("--scene", "--training_data", default="", help="The scene to load. Can be the scene's name or a full path to the training data.")
    parser.add_argument("--mode", default="", const="nerf", nargs="?", choices=["nerf", "sdf", "image", "volume"], help="Mode can be 'nerf', 'sdf', or 'image' or 'volume'. Inferred from the scene if unspecified.")
    parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")

    parser.add_argument("--load_snapshot", default="", help="Load this snapshot before training. recommended extension: .msgpack")
    parser.add_argument("--save_snapshot", default="", help="Save this snapshot after training. recommended extension: .msgpack")

    parser.add_argument("--nerf_compatibility", action="store_true", help="Matches parameters with original NeRF. Can cause slowness and worse results on some scenes.")
    parser.add_argument("--test_transforms", default="", help="Path to a nerf style transforms json from which we will compute PSNR.")
    parser.add_argument("--near_distance", default=-1, type=float, help="set the distance from the camera at which training rays start for nerf. <0 means use ngp default")

    parser.add_argument("--screenshot_transforms", default="", help="Path to a nerf style transforms.json from which to save screenshots.")
    parser.add_argument("--screenshot_frames", nargs="*", help="Which frame(s) to take screenshots of.")
    parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots to.")
    parser.add_argument("--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots.")

    parser.add_argument("--save_mesh", default="", help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.")
    parser.add_argument("--marching_cubes_res", default=256, type=int, help="Sets the resolution for the marching cubes grid.")

    parser.add_argument("--width", "--screenshot_w", type=int, default=0, help="Resolution width of GUI and screenshots.")
    parser.add_argument("--height", "--screenshot_h", type=int, default=0, help="Resolution height of GUI and screenshots.")

    parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
    parser.add_argument("--train", action="store_true", help="If the GUI is enabled, controls whether training starts immediately.")
    parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")

    parser.add_argument("--sharpen", default=0, help="Set amount of sharpening applied to NeRF training images.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "":
        if args.scene in scenes_sdf:
            args.mode = "sdf"
        elif args.scene in scenes_nerf:
            args.mode = "nerf"
        elif args.scene in scenes_image:
            args.mode = "image"
        elif args.scene in scenes_volume:
            args.mode = "volume"
        else:
            raise ValueError("Must specify either a valid '--mode' or '--scene' argument.")

    if args.mode == "sdf":
        mode = ngp.TestbedMode.Sdf
        configs_dir = os.path.join(ROOT_DIR, "configs", "sdf")
        scenes = scenes_sdf
    elif args.mode == "volume":
        mode = ngp.TestbedMode.Volume
        configs_dir = os.path.join(ROOT_DIR, "configs", "volume")
        scenes = scenes_volume
    elif args.mode == "nerf":
        mode = ngp.TestbedMode.Nerf
        configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")
        scenes = scenes_nerf
    elif args.mode == "image":
        mode = ngp.TestbedMode.Image
        configs_dir = os.path.join(ROOT_DIR, "configs", "image")
        scenes = scenes_image

    base_network = os.path.join(configs_dir, "base.json")
    if args.scene in scenes:
        network = scenes[args.scene]["network"] if "network" in scenes[args.scene] else "base"
        base_network = os.path.join(configs_dir, network+".json")
    network = args.network if args.network else base_network
    if not os.path.isabs(network):
        network = os.path.join(configs_dir, network)



    testbed = ngp.Testbed(mode)
    testbed.nerf.sharpen = float(args.sharpen)
    mode2 = ngp.TestbedMode.Nerf
    testbed2 = ngp.Testbed(mode2)
    testbed2.nerf.sharpen = float(args.sharpen)


    if args.mode == "sdf":
        testbed.tonemap_curve = ngp.TonemapCurve.ACES
        testbed2.tonemap_curve = ngp.TonemapCurve.ACESv

    if args.scene:
        scene=args.scene
        if not os.path.exists(args.scene) and args.scene in scenes:
            scene = os.path.join(scenes[args.scene]["data_dir"], scenes[args.scene]["dataset"])
        testbed.load_training_data(scene)
        testbed2.load_training_data(scene)
    if args.load_snapshot:
        print("Loading snapshot ", args.load_snapshot)
        testbed.load_snapshot(args.load_snapshot)
    else:
        testbed.reload_network_from_file(network)
        testbed2.reload_network_from_file(network)


    ref_transforms = {}
    if args.screenshot_transforms: # try to load the given file straight away
        print("Screenshot transforms from ", args.screenshot_transforms)
        with open(args.screenshot_transforms) as f:
            ref_transforms = json.load(f)

    if args.gui:
        # Pick a sensible GUI resolution depending on arguments.
        sw = args.width or 1920
        sh = args.height or 1080
        while sw*sh > 1920*1080*4:
            sw = int(sw / 2)
            sh = int(sh / 2)
        testbed.init_window(sw, sh)

    testbed.shall_train = args.train if args.gui else True
    testbed2.shall_train = args.train if args.gui else True

    testbed.nerf.render_with_camera_distortion = True
    testbed2.nerf.render_with_camera_distortion = True

    network_stem = os.path.splitext(os.path.basename(network))[0]
    if args.mode == "sdf":
        setup_colored_sdf(testbed, args.scene)
        setup_colored_sdf(testbed2, args.scene)


    if args.near_distance >= 0.0:
        print("NeRF training ray near_distance ", args.near_distance)
        testbed.nerf.training.near_distance = args.near_distance
        testbed2.nerf.training.near_distance = args.near_distance

    if args.nerf_compatibility:
        print(f"NeRF compatibility mode enabled")

        # Prior nerf papers accumulate/blend in the sRGB
        # color space. This messes not only with background
        # alpha, but also with DOF effects and the likes.
        # We support this behavior, but we only enable it
        # for the case of synthetic nerf data where we need
        # to compare PSNR numbers to results of prior work.
        testbed.color_space = ngp.ColorSpace.SRGB
        testbed2.color_space = ngp.ColorSpace.SRGB

        # No exponential cone tracing. Slightly increases
        # quality at the cost of speed. This is done by
        # default on scenes with AABB 1 (like the synthetic
        # ones), but not on larger scenes. So force the
        # setting here.
        testbed.nerf.cone_angle_constant = 0
        testbed2.nerf.cone_angle_constant = 0

        # Optionally match nerf paper behaviour and train on a
        # fixed white bg. We prefer training on random BG colors.
        # testbed.background_color = [1.0, 1.0, 1.0, 1.0]
        # testbed.nerf.training.random_bg_color = False


    old_training_step = 0
    n_steps = args.n_steps
    if n_steps < 0:
        n_steps = 100000

    if n_steps > 0:
        with tqdm(desc="Training", total=n_steps, unit="step") as t:
            while testbed.frame():
                if testbed.want_repl():
                    repl(testbed)
                # What will happen when training is done?
                if testbed.training_step >= n_steps:
                    if args.gui:
                        testbed.shall_train = False
                    else:
                        break

                # Update progress bar
                if testbed.training_step < old_training_step or old_training_step == 0:
                    old_training_step = 0
                    t.reset()

                t.update(testbed.training_step - old_training_step)
                t.set_postfix(loss=testbed.loss)
                old_training_step = testbed.training_step

    old_training_step = 0
    n_steps = args.n_steps
    if n_steps < 0:
        n_steps = 100000

    if n_steps > 0:
        with tqdm(desc="Training", total=n_steps, unit="step") as t:
            while testbed2.frame():
                # cropping the useless stuff
                testbed.render_aabb = ngp.BoundingBox([0.049, 0.253, -0.130], [0.946, 0.763, 0.989])
                testbed2.render_aabb = ngp.BoundingBox([0.049, 0.253, -0.130], [0.946, 0.763, 0.989])

                if testbed2.want_repl():
                    repl(testbed2)
                # What will happen when training is done?
                if testbed2.training_step >= n_steps:
                    if args.gui:
                        testbed2.shall_train = False
                    else:
                        break

                # Update progress bar
                if testbed2.training_step < old_training_step or old_training_step == 0:
                    old_training_step = 0
                    t.reset()

                t.update(testbed2.training_step - old_training_step)
                t.set_postfix(loss=testbed2.loss)
                old_training_step = testbed2.training_step



    if args.save_snapshot:
        print("Saving snapshot ", args.save_snapshot)
        testbed.save_snapshot(args.save_snapshot, False)

    # if args.test_transforms:
	#
    #     print("Evaluating test transforms from ", args.test_transforms)
    #     with open(args.test_transforms) as f:
    #         test_transforms = json.load(f)
    #     data_dir=os.path.dirname(args.test_transforms)
    #     totmse = 0
    #     totpsnr = 0
    #     totssim = 0
    #     totcount = 0
    #     minpsnr = 1000
    #     maxpsnr = 0
	#
    #     # Evaluate metrics on black background
    #     testbed.background_color = [0.0, 0.0, 0.0, 1.0]
    #     testbed2.background_color = [0.0, 0.0, 0.0, 1.0]
	#
    #     # Prior nerf papers don't typically do multi-sample anti aliasing.
    #     # So snap all pixels to the pixel centers.
    #     testbed.snap_to_pixel_centers = True
    #     testbed2.snap_to_pixel_centers = True
    #     spp = 8
	#
    #     testbed.nerf.rendering_min_transmittance = 1e-4
    #     testbed2.nerf.rendering_min_transmittance = 1e-4
	#
    #     testbed.fov_axis = 0
    #     testbed2.fov_axis = 0
    #     test_transforms["camera_angle_x"] = 1.0132447905730697
    #     testbed.fov = test_transforms["camera_angle_x"] * 180 / np.pi
    #     testbed.shall_train = False
    #     testbed2.fov = test_transforms["camera_angle_x"] * 180 / np.pi
    #     testbed2.shall_train = False
    #     dir = "image_doublenerf/"
    #     with tqdm(list(enumerate(test_transforms["frames"])), unit="images", desc=f"Rendering test frame") as t:
    #         for i, frame in t:
	#
	#
    #             testbed.set_nerf_camera_matrix(np.matrix(frame["transform_matrix"])[:-1,:])
    #             image = testbed.render(1280, 720, spp, True)
    #             write_image(dir +"testbed1_"+ str(i) + ".png", image)
    #             testbed2.set_nerf_camera_matrix(np.matrix(frame["transform_matrix"])[:-1,:])
    #             image2 = testbed2.render(1280, 720, spp, True)
    #             write_image(dir +"testbed2_"+ str(i) + ".png", image2)
	#
    #             diffimg = np.absolute(image - image2)
	#
    #             diffimg[...,3:4] = 1.0
    #             write_image(dir+"diff" +str(i)+".png", diffimg)
    #             A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
    #             R = np.clip(linear_to_srgb(image2[...,:3]), 0.0, 1.0)
    #             mse = float(compute_error("MSE", A, R))
    #             ssim = float(compute_error("SSIM", A, R))
    #             psnr = mse2psnr(mse)
    #             print("difference for", str(i), "frame is:", np.sum(diffimg), "mse:", mse, "psnr:", psnr, "ssim", ssim)
    #     psnr_avgmse = mse2psnr(totmse/(totcount or 1))
    #     psnr = totpsnr/(totcount or 1)
    #     ssim = totssim/(totcount or 1)
    #     print(f"PSNR={psnr} [min={minpsnr} max={maxpsnr}] SSIM={ssim}")
	#
    # if args.save_mesh:
    #     res = args.marching_cubes_res or 256
    #     print(f"Generating mesh via marching cubes and saving to {args.save_mesh}. Resolution=[{res},{res},{res}]")
    #     testbed.compute_and_save_marching_cubes_mesh(args.save_mesh, [res, res, res])
	#
    # if args.width:
    #     if ref_transforms:
    #         testbed.fov_axis = 0
    #         testbed.fov = ref_transforms["camera_angle_x"] * 180 / np.pi
    #         if not args.screenshot_frames:
    #             args.screenshot_frames = range(len(ref_transforms["frames"]))
    #         print(args.screenshot_frames)
    #         for idx in args.screenshot_frames:
    #             f = ref_transforms["frames"][int(idx)]
    #             cam_matrix = f["transform_matrix"]
    #             testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1,:])
    #             outname = os.path.join(args.screenshot_dir, os.path.basename(f["file_path"]))
	#
    #             # Some NeRF datasets lack the .png suffix in the dataset metadata
    #             if not os.path.splitext(outname)[1]:
    #                 outname = outname + ".png"
	#
    #             print(f"rendering {outname}")
    #             image = testbed.render(args.width or int(ref_transforms["w"]), args.height or int(ref_transforms["h"]), args.screenshot_spp, True)
    #             os.makedirs(os.path.dirname(outname), exist_ok=True)
    #             write_image(outname, image)
    #     elif args.screenshot_dir:
    #         outname = os.path.join(args.screenshot_dir, args.scene + "_" + network_stem)
    #         print(f"Rendering {outname}.png")
    #         image = testbed.render(args.width, args.height, args.screenshot_spp, True)
    #         if os.path.dirname(outname) != "":
    #             os.makedirs(os.path.dirname(outname), exist_ok=True)
    #         write_image(outname + ".png", image)
	#
	#
	#
