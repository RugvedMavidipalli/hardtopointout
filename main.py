# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Tutorial to show how to use Spot's arm.
"""
from __future__ import print_function
import math
import argparse
import sys
import time
import cv2
import os
import numpy as np
import bosdyn.api.gripper_command_pb2
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.client.docking import blocking_undock, blocking_dock_robot
from bosdyn.api import arm_command_pb2, geometry_pb2, trajectory_pb2
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b, VISION_FRAME_NAME
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, VISION_FRAME_NAME, BODY_FRAME_NAME, get_se2_a_tform_b
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.api import image_pb2
from bosdyn.api import gripper_camera_param_pb2
from bosdyn import geometry
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
import json
from colmap2nerf import do_system, colmap_main
from mp_run import run_testbed
from multiprocessing import Process, Manager
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import plotly.graph_objects as go

TRANSFORM = []
CURRENT_POSITION = (0, 0, 0)
CURRENT_LOC_NO = 0
with open("robotLocation.json", "r") as outfile:
    robotLocations = json.load(outfile)
with open("cameraLocation.json", "r") as outfile:
    cameraLocations = json.load(outfile)

def main(config):

    """A simple example of using the Boston Dynamics API to command Spot's arm."""

    # See hello_spot.py for an explanation of these lines.
    bosdyn.client.util.setup_logging(config.verbose)

    sdk = bosdyn.client.create_standard_sdk('HelloSpotClient')
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    #image stuff
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), "Robot requires an arm to run this example."

    # Verify the robot is not estopped and that an external application has registered and holds
    # an estop endpoint.
    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                    "such as the estop SDK example, to configure E-Stop."

    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    #image client for taking image
    image_client = robot.ensure_client(ImageClient.default_service_name)


    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Now, we are ready to power on the robot. This call will block until the power
        # is on. Commands would fail if this did not happen. We can also check that the robot is
        # powered at any point.
        robot.logger.info("Powering on robot... This may take a several seconds.")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."
        robot.logger.info("Robot powered on.")

        # Tell the robot to stand up. The command service is used to issue commands to a robot.
        # The set of valid commands for a robot depends on hardware configuration. See
        # SpotCommandHelper for more detailed examples on command building. The robot
        # command service requires timesync between the robot and the client.
        robot.logger.info("Commanding robot to stand...")
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        #blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot undocking...\nCLEAR AREA in front of docking station.")
        blocking_undock(robot)

        robot.logger.info("Robot undocked and standing")
        time.sleep(3)

        robotLocation = robotLocations['0']
        absolute_move(robotLocation[0], robotLocation[1], math.radians(robotLocation[2]), VISION_FRAME_NAME,
                      command_client,
                      robot_state_client)

        # clear data directory and remake
        if os.path.exists(config.images_path):
            do_system(f'rm -r {config.images_path}')
        if os.path.exists(config.output_path):
            do_system(f'rm -r {config.output_path}')
        do_system(f'mkdir -p {config.images_path}')
        do_system(f'mkdir -p {config.output_path}')

        # take initial images
        cameralist = list(range(0, 11, 2))

        for cameraSpot in cameralist:
            goto_location_take_image(config, robot, robot_state_client, image_client, command_client, cameraLocations[cameraSpot])

        # run colmap (or cheat)
        if config.cheat_images:
            do_system(f'cp {config.cheat_images_path}/transforms.json {config.images_path}/')
        else:
            colmap_main(True, config.images_path, config.aabb_scale)

        double_nerf = False

        if double_nerf:
            manager = Manager()
            testbed_data1 = manager.dict()
            testbed_data2 = manager.dict()

            testbed_data1['train'] = False
            testbed_data2['train'] = False
            testbed_data1['render'] = False
            testbed_data2['render'] = False
            testbed_data1['load_new_data'] = False
            testbed_data2['load_new_data'] = False

            p1 = Process(target=run_testbed, args=['../../configs/nerf', f'{config.images_path}/transforms.json', f'{config.output_path}', testbed_data1, 0, True])
            p1.start()
            p2 = Process(target=run_testbed, args=['../../configs/nerf', f'{config.images_path}/transforms.json', f'{config.output_path}', testbed_data2, 1, False])
            p2.start()
            testbed_data1['train'] = True
            testbed_data2['train'] = True
            testbed_data1['load_new_data'] = False
            testbed_data2['load_new_data'] = False





            for i in range(10):
                time.sleep(5)
                testbed_data1['render'] = True
                testbed_data2['render'] = True

                with open(f'{config.images_path}/transforms.json', 'r') as f:
                    transforms = json.load(f)
                vals = np.zeros(len(transforms['frames']))
                locs = np.zeros((len(transforms['frames']), 3))
                while testbed_data1['render'] or testbed_data2['render']:
                    pass
                # fig, ax = plt.subplots(16, 6, squeeze=True)
                for frame in transforms['frames']:
                    index = int(frame['file_path'][frame['file_path'].rfind('/') + 1:-4])
                    locs[index, :] = np.array(frame['transform_matrix'])[:3, 3]
                    image1 = np.load(f"{config.output_path}/{frame['file_path'][:-4]}_0.npy")
                    image2 = np.load(f"{config.output_path}/{frame['file_path'][:-4]}_1.npy")
                    # ax[int(index / 6)][index % 6].imshow(np.sum(image1 - image2, 2) ** 2)
                    vals[index] = np.mean((image1 - image2) ** 2)
                # plt.show()
                vals -= np.min(vals)
                vals /= np.max(vals)

                viridis = cm.get_cmap('viridis', 12)
                color = viridis(vals)
                color[0] = (1, 0, 0, 1)

                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                plot_point_mat(ax, locs, color, 'o')
                ax.set_box_aspect([1, 1, 1])  # IMPORTANT - this is the new, key line
                # ax.set_proj_type('ortho') # OPTIONAL - default is perspective (shown in image above)
                set_axes_equal(ax)  # IMPORTANT - this is also required
                plt.show()

                next_pose = np.argmax(vals)
                goto_location_take_image(config, robot, robot_state_client, image_client, command_client,
                                         cameraLocations[next_pose])

                # run colmap (or cheat)
                if not config.cheat_images:
                    colmap_main(False, config.images_path, config.aabb_scale)

                testbed_data1['train'] = False
                testbed_data1['load_new_data'] = True
                testbed_data2['train'] = False
                testbed_data2['load_new_data'] = True



            # take more pictures
            cameralist = list(range(22, 33, 3))

            for cameraSpot in cameralist:
                goto_location_take_image(config, robot, robot_state_client, image_client, command_client,
                                         cameraLocations[cameraSpot])

            # run colmap (or cheat)
            if not config.cheat_images:
                colmap_main(False, config.images_path, config.aabb_scale)

            testbed_data1['train'] = False
            testbed_data1['load_new_data'] = True
            testbed_data1['train'] = True

            while input() != 'e':
                pass
        else:
            manager = Manager()
            testbed_data1 = manager.dict()

            testbed_data1['train'] = False
            testbed_data1['render'] = False
            testbed_data1['load_new_data'] = False
            testbed_data1['render_depth'] = False

            p1 = Process(target=run_testbed, args=['../../configs/nerf', f'{config.images_path}/transforms.json', f'{config.output_path}', testbed_data1, 0])
            p1.start()
            testbed_data1['train'] = True

            time.sleep(10)

            testbed_data1['render_depth'] = True

            while testbed_data1['render_depth']:
                pass
            for count in range(10):
                def load_transforms(transforms_path):
                    with open(transforms_path, 'r') as f:
                        json_data = json.load(f)
                    extrinsics = {
                        int(frame['file_path'][frame['file_path'].rfind('/') + 1:-4]): np.array(frame['transform_matrix'])
                        for frame in json_data['frames']}
                    intrinsics = np.array(
                        [[json_data['fl_x'], 0, json_data['cx']], [0, json_data['fl_y'], json_data['cy']], [0, 0, 1]])
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
                    for px in range(0, depth.shape[1], 25):
                        for py in range(0, depth.shape[0], 25):
                            camera_xyz = (intrinsics_inverse @ np.array([px, py, 1])) * (
                                        1 / np.exp(0.6931471805599453 * -5)) * -depth[py, px, 0]
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
                        if py < h and px < w and py >= 0 and px >= 0 and camera_xyz[2] > out_depth[py, px] and camera_xyz[
                            2] < -0.1:
                            out_depth[py, px] = camera_xyz[2]
                            out_color[py, px] = color
                            out_qn[py, px] = 1 if camera_xyz[2] > camera_camera_center[2] else 0
                    return out_depth, out_color, out_qn

                intrinsics, input_transforms = load_transforms(config.images_path + '/transforms.json')
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
                        new_xyz, new_color, new_quasinormals = backproject(intrinsics, input_transforms[i], depths[i],
                                                                           colors[i])
                        if xyz is None:
                            xyz = new_xyz
                            quasinormals = new_quasinormals
                        else:
                            xyz = np.concatenate([xyz, new_xyz], 0)
                            quasinormals = np.concatenate([quasinormals, new_quasinormals], 0)
                        color += new_color
                locs = np.zeros((len(keys), 3))
                loc_color = [0] * len(keys)
                for key in keys:
                    locs[key, :] = input_transforms[key][:3, 3]
                    rendered_depth, rendered_color, rendered_qn = project(intrinsics, 30, 54, 720, 1280,
                                                                          input_transforms[key], xyz, quasinormals, color)
                    mask = rendered_depth != -10
                    all = np.sum(mask)
                    good = np.sum(rendered_qn[mask])
                    print(good / all)
                    if key in depths:
                        loc_color[key] = 2
                    else:
                        loc_color[key] = good / all

                scores = np.array([score for score in loc_color])
                new_loc = np.argmin(scores)

                viridis = cm.get_cmap('viridis', 12)
                new_loc_color = [x for x in viridis(np.array(loc_color))]
                new_loc_color = [x if y != 2 else (1, 0, 0, 1) for x, y in zip(new_loc_color, loc_color)]
                new_loc_color[new_loc] = (0, 1, 1, 1)
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
                print(len(color))
                testbed_data1['render_depth'] = True
                index = 0
                while testbed_data1['render_depth']:
                    goto_location_take_image(config, robot, robot_state_client, image_client, command_client,
                                         cameraLocations[np.argsort(scores)[index]])
                    index += 1
                testbed_data1['load_new_data'] = True

            # return to location 0 and then dock
            goto_robot_location(robot_state_client, command_client, 0)
            absolute_move(1, 0, 0, VISION_FRAME_NAME, command_client, robot_state_client)
            blocking_dock_robot(robot, 521)
            robot.power_off(cut_immediately=False, timeout_sec=20)
            assert not robot.is_powered_on(), "Robot power off failed."
            robot.logger.info("Robot safely powered off.")
            print(TRANSFORM)

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

def goto_location_take_image(config, robot, robot_state_client, image_client, command_client, cameralocation):
    new_location = int(cameralocation['robotLocation'])
    goto_robot_location(robot_state_client, command_client, new_location)
    imageName = cameralocation['imageNo']
    x, y ,z = cameralocation['gripperXYZ']
    yaw = cameralocation['yaw']
    pitch = cameralocation['pitch']
    [qx, qy, qz, qw] = euler_to_quaternion(yaw, pitch, 0)
    arm_move(x, y, z, qw, qx, qy, qz, robot,
                robot_state_client, command_client)
    time.sleep(2)

    image_path = f'{config.images_path}/{imageName}.jpg'
    take_image(image_client, image_path)
    if config.cheat_images:
        do_system(f'cp {config.cheat_images_path}/{imageName}.jpg {image_path}')

def goto_robot_location(robot_state_client, command_client, new_location):
    print(f'goto_robot_location({new_location})')
    global CURRENT_LOC_NO
    if new_location == CURRENT_LOC_NO:
        return
    else:
        stow = RobotCommandBuilder.arm_stow_command()

        # Issue the command via the RobotCommandClient
        stow_command_id = command_client.robot_command(stow)


        block_until_arm_arrives(command_client, stow_command_id, 3.0)
        if (new_location - CURRENT_LOC_NO + 8) % 8 < (CURRENT_LOC_NO - new_location + 8) % 8:
            for i in range(2, (new_location - CURRENT_LOC_NO + 8) % 8, 2):
                robotLocation = robotLocations[str((CURRENT_LOC_NO + i) % 8)]
                absolute_move(robotLocation[0], robotLocation[1], math.radians(robotLocation[2]), VISION_FRAME_NAME,
                              command_client,
                              robot_state_client)
                CURRENT_LOC_NO = (CURRENT_LOC_NO + i) % 8
            if new_location != CURRENT_LOC_NO:
                robotLocation = robotLocations[str(new_location)]
                absolute_move(robotLocation[0], robotLocation[1], math.radians(robotLocation[2]), VISION_FRAME_NAME,
                              command_client,
                              robot_state_client)
                CURRENT_LOC_NO = new_location
        else:
            for i in range(2, (CURRENT_LOC_NO - new_location + 8) % 8, 2):
                robotLocation = robotLocations[str((CURRENT_LOC_NO - i) % 8)]
                absolute_move(robotLocation[0], robotLocation[1], math.radians(robotLocation[2]), VISION_FRAME_NAME,
                              command_client,
                              robot_state_client)
                CURRENT_LOC_NO = (CURRENT_LOC_NO - i) % 8
            if new_location != CURRENT_LOC_NO:
                robotLocation = robotLocations[str(new_location)]
                absolute_move(robotLocation[0], robotLocation[1], math.radians(robotLocation[2]), VISION_FRAME_NAME,
                              command_client,
                              robot_state_client)
                CURRENT_LOC_NO = new_location



def take_images(robot, robot_state_client, command_client, image_client, start_index):
    # xyz of the robot girpper relative to the torso
    x, y, z = 0.75, 0, 0.25
    # Rotation as a quaternion
    [qx, qy, qz, qw] = euler_to_quaternion(0, 0, 0)
    # move the arm based on the position and quaternion
    arm_move(x, y, z, qw, qx, qy, qz, robot,
             robot_state_client, command_client)

    start_y, finish_y, num_shot = 0.35, -0.35, 11
    ys= np.linspace(start_y, finish_y, num_shot)

    # x, z remains as constants for now
    x, z = 0.85, -0.15
    pitch = 30
    # x, z = 1, 0.14
    # pitch = 65
    #when arm is out, the relative distance on xaxis is around 60cm
    distance_to_bottle = 0.6

    for y in ys:
        yaw_angle = -np.arctan(y / distance_to_bottle) / np.pi * 180
        [qx, qy, qz, qw] = euler_to_quaternion(yaw_angle, pitch, 0)
        arm_move(x, y, z, qw, qx, qy, qz, robot,
                robot_state_client, command_client)
        time.sleep(3)
        take_image(image_client, name=str(start_index))
        TRANSFORM.append({"image_No": start_index, "current pose": CURRENT_POSITION,
                          "gripperXYZYawPitch":(x,y,z,yaw_angle, pitch)})
        start_index += 1


    stow = RobotCommandBuilder.arm_stow_command()

    # Issue the command via the RobotCommandClient
    stow_command_id = command_client.robot_command(stow)

    robot.logger.info("Stow command issued.")
    block_until_arm_arrives(command_client, stow_command_id, 3.0)
    return start_index

def walk_to_other_side(command_client, robot_state_client):
    try:
        # move the robot to the side
        dy = 0.5
        # 1.1 is the body length.
        distance_to_obj = 0.8
        dx = distance_to_obj * 2 + 1.1

        dyaw = 180
        relative_move(0, dy, math.radians(0), ODOM_FRAME_NAME,
                      command_client, robot_state_client)
        # move the robot forward
        relative_move(dx, 0, math.radians(0), ODOM_FRAME_NAME,
                      command_client, robot_state_client)
        # turn the robot
        relative_move(0, 0, math.radians(dyaw), ODOM_FRAME_NAME,
                      command_client, robot_state_client)
        # move the robot back to the center
        relative_move(0, dy, math.radians(0), ODOM_FRAME_NAME,
                      command_client, robot_state_client)

    finally:
        # Send a Stop at the end, regardless of what happened.
        command_client.robot_command(RobotCommandBuilder.stop_command())

def walk_to_left_side(command_client, robot_state_client):
    try:
        # move the robot to the side

        # 1.1 is the body length.
        distance_to_obj = 0.85
        dx = (distance_to_obj * 2 + 1.1)/2
        dy = dx

        dyaw = -90
        relative_move(0, dy, math.radians(0), ODOM_FRAME_NAME,
                      command_client, robot_state_client)
        # move the robot forward
        relative_move(dx, 0, math.radians(0), ODOM_FRAME_NAME,
                      command_client, robot_state_client)
        # turn the robot
        relative_move(0, 0, math.radians(dyaw), ODOM_FRAME_NAME,
                      command_client, robot_state_client)


    finally:
        # Send a Stop at the end, regardless of what happened.
        command_client.robot_command(RobotCommandBuilder.stop_command())

def block_until_arm_arrives_with_prints(robot, command_client, cmd_id):
    """Block until the arm arrives at the goal and print the distance remaining.
        Note: a version of this function is available as a helper in robot_command
        without the prints.
    """
    while True:
        feedback_resp = command_client.robot_command_feedback(cmd_id)
        robot.logger.info(
            'Distance to go: ' +
            '{:.2f} meters'.format(feedback_resp.feedback.synchronized_feedback.arm_command_feedback
                                   .arm_cartesian_feedback.measured_pos_distance_to_goal) +
            ', {:.2f} radians'.format(
                feedback_resp.feedback.synchronized_feedback.arm_command_feedback.
                arm_cartesian_feedback.measured_rot_distance_to_goal))

        if feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback.status == arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_COMPLETE:
            robot.logger.info('Move complete.')
            break
        time.sleep(0.1)

def arm_move(x,y,z,qw,qx,qy,qz,robot,robot_state_client,command_client):
    # Move the arm to a spot in front of the robot, and open the gripper.
    # Make the arm pose RobotCommand
    # Build a position to move the arm to (in meters, relative to and expressed in the gravity aligned body frame).

    hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)
    flat_body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)
    flat_body_T_hand = geometry_pb2.SE3Pose(position=hand_ewrt_flat_body,
                                            rotation=flat_body_Q_hand)

    robot_state = robot_state_client.get_robot_state()
    odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                     ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

    odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_obj(flat_body_T_hand)
    # duration in seconds
    seconds = 2
    arm_command = RobotCommandBuilder.arm_pose_command(
        odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, odom_T_hand.rot.w, odom_T_hand.rot.x,
        odom_T_hand.rot.y, odom_T_hand.rot.z, ODOM_FRAME_NAME, seconds)

    # Make the open gripper RobotCommand
    gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)

    # Combine the arm and gripper commands into one RobotCommand
    command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)

    # Send the request
    cmd_id = command_client.robot_command(command)
    robot.logger.info('Moving arm to a position.')

    # Wait until the arm arrives at the goal.
    block_until_arm_arrives_with_prints(robot, command_client, cmd_id)

def relative_move(dx, dy, dyaw, frame_name, robot_command_client, robot_state_client, stairs=False):
    global CURRENT_POSITION
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

    # Build the transform for where we want the robot to be relative to where the body currently is.
    body_tform_goal = math_helpers.SE2Pose(x=dx, y=dy, angle=dyaw)
    # We do not want to command this goal in body frame because the body will move, thus shifting
    # our goal. Instead, we transform this offset to get the goal position in the output frame
    # (which will be either odom or vision).
    out_tform_body = get_se2_a_tform_b(transforms, frame_name, BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal

    # Command the robot to go to the goal point in the specified frame. The command will stop at the
    # new position.
    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
        frame_name=frame_name, params=RobotCommandBuilder.mobility_params(stair_hint=stairs))
    end_time = 10.0
    cmd_id = robot_command_client.robot_command(lease=None, command=robot_cmd,
                                                end_time_secs=time.time() + end_time)
    # Wait until the robot has reached the goal.
    while True:
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
        if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("Failed to reach the goal")
            return False
        traj_feedback = mobility_feedback.se2_trajectory_feedback
        if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
            print("Arrived at the goal.")
            return True
        time.sleep(1)

    return True

def absolute_move(x, y, yaw, frame_name, robot_command_client, robot_state_client, stairs=False):


    # Command the robot to go to the goal point in the specified frame. The command will stop at the
    # new position.
    global CURRENT_POSITION
    CURRENT_POSITION = (x, y , yaw)
    mobility_params = set_mobility_params()
    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=x, goal_y=y, goal_heading=yaw,
        frame_name=frame_name, params=mobility_params)
    end_time = 20.0
    cmd_id = robot_command_client.robot_command(lease=None, command=robot_cmd,
                                                end_time_secs=time.time() + end_time)
    # Wait until the robot has reached the goal.
    while True:
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
        if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("Failed to reach the goal")
            return False
        traj_feedback = mobility_feedback.se2_trajectory_feedback
        if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
            print("Arrived at the goal.")
            return True
        time.sleep(1)

    return True

def take_image(image_client, path):
    pixel_format = image_pb2.Image.PIXEL_FORMAT_RGB_U8
    image_request = build_image_request("hand_color_image", pixel_format=pixel_format)
    image_responses = image_client.get_image([image_request])
    for image in image_responses:
        num_bytes = 3
        dtype = np.uint8
        img = np.frombuffer(image.shot.image.data, dtype=dtype)
        img = cv2.imdecode(img, -1)
        image_saved_path = path
        #image_saved_path = image_saved_path.replace("/", '')
        cv2.imwrite(image_saved_path, img)

def set_mobility_params():
    max_x_vel, max_y_vel, max_ang_vel = 0.5, 0.5, 1.0
    body_control = set_default_body_control()

    speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(
            linear=Vec2(x=max_x_vel, y=max_y_vel), angular=max_ang_vel))

    mobility_params = spot_command_pb2.MobilityParams(
                vel_limit=speed_limit, body_control=body_control,
                locomotion_hint=spot_command_pb2.HINT_AUTO)

    return mobility_params

def set_default_body_control():
    """Set default body control params to current body position"""
    footprint_R_body = geometry.EulerZXY()
    position = geometry_pb2.Vec3(x=0.0, y=0.0, z=0.0)
    rotation = footprint_R_body.to_quaternion()
    pose = geometry_pb2.SE3Pose(position=position, rotation=rotation)
    point = trajectory_pb2.SE3TrajectoryPoint(pose=pose)
    traj = trajectory_pb2.SE3Trajectory(points=[point])
    return spot_command_pb2.BodyControlParams(base_offset_rt_footprint=traj)

def euler_to_quaternion(yaw, pitch, roll):
    yaw = degree_to_radiance(yaw)
    pitch = degree_to_radiance(pitch)
    roll = degree_to_radiance(roll)
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]

def degree_to_radiance(degree):
    return degree / 360 * 2 * np.pi

def pixel_format_string_to_enum(enum_string):
    return dict(image_pb2.Image.PixelFormat.items()).get(enum_string)
