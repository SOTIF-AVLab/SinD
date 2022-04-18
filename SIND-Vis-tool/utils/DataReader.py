import pandas as pd
import os
import numpy as np


def read_tracks_all(path):

    Veh_path = os.path.join(path, 'Veh_smoothed_tracks.csv')
    Ped_path = os.path.join(path, 'Ped_smoothed_tracks.csv')
    veh_df = pd.read_csv(Veh_path)
    veh_state_name = veh_df.columns.tolist()
    assert veh_state_name == ['track_id', 'frame_id', 'timestamp_ms', 'agent_type', 'x', 'y', 'vx', 'vy', 'yaw_rad',
                         'heading_rad', 'length', 'width', 'ax', 'ay', 'v_lon', 'v_lat', 'a_lon', 'a_lat']

    Veh_tracks_dict = {}
    Veh_tracks_group = veh_df.groupby(["track_id"], sort=False)
    for track_id, group in Veh_tracks_group:
        track = {}
        track_dict = group.to_dict(orient="list")
        for key, value in track_dict.items():
            if key in ["track_id", "agent_type"]:

                track[key] = value[0]

            else:
                track[key] = np.array(value)

        track["center"] = np.stack([track["x"], track["y"]], axis=-1)
        track["bbox"], track["triangle"] = calculate_rot_bboxes_and_triangle(track["x"], track["y"],
                                                                    track["length"], track["width"],
                                                                    track["yaw_rad"])
        Veh_tracks_dict[track_id] = track #track["yaw_rad"]

    ped_df = pd.read_csv(Ped_path)
    statename = ped_df.columns.tolist()
    assert statename == ['track_id', 'frame_id', 'timestamp_ms', 'agent_type', 'x', 'y', 'vx', 'vy', 'ax', 'ay']
    Ped_tracks_dict = {}
    Ped_tracks_group = ped_df.groupby(["track_id"], sort=False)
    for track_id, group in Ped_tracks_group:
        track = {}
        track_dict = group.to_dict(orient="list")
        for key, value in track_dict.items():
            if key in ["track_id", "agent_type"]:
                track[key] = value[0]
            else:
                track[key] = np.array(value)
        track["center"] = np.stack([track["x"], track["y"]], axis=-1)
        track["bbox"], _ = calculate_rot_bboxes_and_triangle(track["x"], track["y"])

        Ped_tracks_dict[track_id] = track

    return Veh_tracks_dict, Ped_tracks_dict



def read_light(path, maxframe):

    df_light = pd.read_csv(path)
    light_dict = {}
    memory = 0
    frame = 0
    flag = 0

    for row in df_light.itertuples():
        if row[1] < frame:
            memory = row[3:]
            continue
        while frame < row[1]:
            light_dict[frame] = memory
            frame += 1
            if frame > maxframe + 100:
                flag = 1
                break
        memory = row[3:]
        if flag == 1:
            break

    return light_dict


def read_tracks_meta(path):

    tracks_meta_df = pd.read_csv(path, index_col="trackId")

    tracks_meta_dict = tracks_meta_df.to_dict("index")

    return tracks_meta_dict



def calculate_rot_bboxes_and_triangle(center_points_x, center_points_y, length=0.5, width=0.5, rotation=0):
    """
    Calculate bounding box vertices and triangle vertices from centroid, width and length.

    :param centroid: center point of bbox
    :param length: length of bbox
    :param width: width of bbox
    :param rotation: rotation of main bbox axis (along length)
    :return:
    """

    centroid = np.array([center_points_x, center_points_y]).transpose()#(n, 2)

    centroid = np.array(centroid)
    if centroid.shape == (2,):
        centroid = np.array([centroid])

    # Preallocate
    data_length = centroid.shape[0]
    rotated_bbox_vertices = np.empty((data_length, 4, 2))

    # Calculate rotated bounding box vertices
    rotated_bbox_vertices[:, 0, 0] = -length / 2
    rotated_bbox_vertices[:, 0, 1] = -width / 2

    rotated_bbox_vertices[:, 1, 0] = length / 2
    rotated_bbox_vertices[:, 1, 1] = -width / 2

    rotated_bbox_vertices[:, 2, 0] = length / 2
    rotated_bbox_vertices[:, 2, 1] = width / 2

    rotated_bbox_vertices[:, 3, 0] = -length / 2
    rotated_bbox_vertices[:, 3, 1] = width / 2

    for i in range(4):
        th, r = cart2pol(rotated_bbox_vertices[:, i, :])
        rotated_bbox_vertices[:, i, :] = pol2cart(th + rotation, r).squeeze()
        rotated_bbox_vertices[:, i, :] = rotated_bbox_vertices[:, i, :] + centroid #(n, 4, 2)

    # Calculate triangle vertices
    triangle_factor = 0.75

    a = rotated_bbox_vertices[:, 3, :] + (
                (rotated_bbox_vertices[:, 2, :] - rotated_bbox_vertices[:, 3, :]) * triangle_factor)
    b = rotated_bbox_vertices[:, 0, :] + (
                (rotated_bbox_vertices[:, 1, :] - rotated_bbox_vertices[:, 0, :]) * triangle_factor)
    c = rotated_bbox_vertices[:, 2, :] + ((rotated_bbox_vertices[:, 1, :] - rotated_bbox_vertices[:, 2, :]) * 0.5)

    triangle_array = np.array([a, b, c]).swapaxes(0, 1) #(3, n, 2)


    return rotated_bbox_vertices, triangle_array


def cart2pol(cart):
    """
    Transform cartesian to polar coordinates.
    :param cart: Nx2 ndarray
    :return: 2 Nx1 ndarrays
    """
    if cart.shape == (2,):
        cart = np.array([cart])

    x = cart[:, 0]
    y = cart[:, 1]

    th = np.arctan2(y, x)
    r = np.sqrt(np.power(x, 2) + np.power(y, 2))
    return th, r


def pol2cart(th, r):
    """
    Transform polar to cartesian coordinates.
    :param th: Nx1 ndarray
    :param r: Nx1 ndarray
    :return: Nx2 ndarray
    """

    x = np.multiply(r, np.cos(th))
    y = np.multiply(r, np.sin(th))

    cart = np.array([x, y]).transpose()
    return cart

