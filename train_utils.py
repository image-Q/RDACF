import torch
from torch.nn import functional as F
import numpy as np
import random
from shapely.geometry import LineString, Point, Polygon
from shapely.affinity import affine_transform, rotate
import matplotlib.pyplot as plt


LAMBDA_KINEMATIC = 0.1
LAMBDA_COLLISION = 0.1
DELTA_T = 0.1
SAFE_DISTANCE = 2.0



def data_process(data, batch_size):
    env_input = {'ego_state': [], 'ego_map': [], 'neighbors_state': [], 'neighbors_map': []}
    batch_plan = []
    batch_ground_truth = []

    # process a mini batch
    while len(batch_plan) < batch_size:
        id = random.choice(list(data.keys()))  # sample a episode
        episode_data = data[id]
        timesteps = list(episode_data.keys())
        ego_id = id.split(';')[-1]
        t = random.choice(timesteps[:len(timesteps) - 15])  # sample a timestep

        neighbors = []
        neighbors_map = []
        neighbors_ground_truth = []
        agent_list = []
        current = episode_data[t][ego_id]['state'].copy()

        # add agents
        for agent in episode_data[t].keys():
            if agent == ego_id:
                ego = get_history(episode_data, agent, t)
                ego = traj_transform_to_ego_frame(ego, current)
                env_input['ego_state'].append(ego)

                plan = get_future(episode_data, agent, t)
                plan = traj_transform_to_ego_frame(plan, current)
                batch_plan.append(plan)

                map = episode_data[t][agent]['map'].copy()
                map = map_transform_to_ego_frame(map, current)
                env_input['ego_map'].append(map)

            elif episode_data[t][agent]['map'] is not None:
                agent_list.append(agent)
                neighbor = get_history(episode_data, agent, t)
                neighbor = traj_transform_to_ego_frame(neighbor, current)
                neighbors.append(neighbor)

                gt = get_future(episode_data, agent, t)
                gt = traj_transform_to_ego_frame(gt, current)
                neighbors_ground_truth.append(gt)

                map = episode_data[t][agent]['map'].copy()
                map = map_transform_to_ego_frame(map, current)
                neighbors_map.append(map)

            else:
                continue

        # pad missing agents
        if len(agent_list) < 5:
            neighbors.extend([np.zeros(shape=(11, 5))] * (5 - len(agent_list)))
            neighbors_map.extend([np.zeros(shape=(3, 51, 4))] * (5 - len(agent_list)))
            neighbors_ground_truth.extend([np.zeros(shape=(30, 5))] * (5 - len(agent_list)))

        # add to dict
        env_input['neighbors_state'].append(np.stack(neighbors))
        env_input['neighbors_map'].append(np.stack(neighbors_map))
        batch_ground_truth.append(np.stack(neighbors_ground_truth))

    for k, v in env_input.items():
        env_input[k] = np.stack(v)

    plan = np.stack(batch_plan)
    ground_truth = np.stack(batch_ground_truth)

    return env_input, plan, ground_truth


def get_history(buffer, id, timestep):
    history = np.zeros(shape=(11, 4))
    timesteps = range(timestep + 1)
    idx = -1

    for t in reversed(timesteps):
        if id not in buffer[t].keys() or idx < -11:
            break

        history[idx] = buffer[t][id]['state'].copy()
        idx -= 1

    return history


def get_future(buffer, id, timestep):
    future = np.zeros(shape=(30, 4))
    timesteps = range(timestep + 1, timestep + 31)

    for idx, t in enumerate(timesteps):
        if id not in buffer[t].keys():
            break

        future[idx] = buffer[t][id]['state'].copy()

    return future


def wrap_to_pi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def traj_transform_to_ego_frame(traj, ego_current):
    line = LineString(traj[:, :2])
    center, angle = ego_current[:2], ego_current[2]
    line_offset = affine_transform(line, [1, 0, 0, 1, -center[0], -center[1]])
    line_rotate = rotate(line_offset, -angle, origin=(0, 0), use_radians=True)
    line_rotate = np.array(line_rotate.coords)
    line_rotate[traj[:, :2] == 0] = 0
    heading = wrap_to_pi(traj[:, 2] - angle)
    v = traj[:, 3]
    v_x, v_y = v * np.cos(heading), v * np.sin(heading)
    traj = np.column_stack((line_rotate, heading, v_x, v_y))

    return traj


def map_transform_to_ego_frame(map, ego_current):
    center, angle = ego_current[:2], ego_current[2]

    for i in range(map.shape[0]):
        if map[i, 0, 0] != 0:
            line = LineString(map[i, :, :2])
            line = affine_transform(line, [1, 0, 0, 1, -center[0], -center[1]])
            line = rotate(line, -angle, origin=(0, 0), use_radians=True)
            line = np.array(line.coords)
            line[map[i, :, :2] == 0] = 0
            heading = wrap_to_pi(map[i, :, 2] - angle)
            speed_limit = map[i, :, 3]
            map[i] = np.column_stack((line, heading, speed_limit))

    return map


def predictor_train_step(model, data, batch_size, device):
    env_input, plan, ground_truth = data_process(data, batch_size)
    env_input = {key: torch.as_tensor(_obs).float().to(device) for (key, _obs) in env_input.items()}
    plan = torch.as_tensor(plan).float().to(device)             # [B, T, 5] 自车未来轨迹（在 ego 当前坐标系）
    ground_truth = torch.as_tensor(ground_truth).float().to(device)  # [B, N, T, 5] 邻车未来 GT

    prediction = model(env_input, plan)                         # [B, N, T, 3]


    regression_loss = calculate_loss(prediction, ground_truth)


    kin_loss = kinematic_consistency_loss(prediction, ground_truth, delta_t=DELTA_T)

    coll_loss = collision_regularizer(prediction, ground_truth, plan, safe_distance=SAFE_DISTANCE)

    loss = regression_loss \
           + LAMBDA_KINEMATIC * kin_loss \
           + LAMBDA_COLLISION * coll_loss

    ade, fde = calculate_metrics(prediction, ground_truth)

    return loss, ade, fde



def calculate_loss(prediction, ground_truth):
    # ground_truth: [B, N, T, 5] -> 只用前三维 (x, y, heading)
    valid = torch.ne(ground_truth, 0)[:, :, :, 0, None]   # [B, N, T, 1]
    regression_loss = F.smooth_l1_loss(prediction * valid, ground_truth[..., :3])

    return regression_loss



def wrap_to_pi_torch(theta):

    return (theta + np.pi) % (2 * np.pi) - np.pi


def kinematic_consistency_loss(prediction, ground_truth, delta_t=0.1):

    B, N, T, _ = prediction.shape


    valid = torch.ne(ground_truth, 0)[:, :, :, 0]         # [B, N, T]
    if T <= 1:
        return prediction.new_tensor(0.0)


    pos_pred = prediction[..., :2]            # [B,N,T,2]
    pos_gt = ground_truth[..., :2]           # [B,N,T,2]


    dv_pred = pos_pred[:, :, 1:, :] - pos_pred[:, :, :-1, :]   # [B,N,T-1,2]
    dv_gt = pos_gt[:, :, 1:, :] - pos_gt[:, :, :-1, :]         # [B,N,T-1,2]

    speed_pred = torch.norm(dv_pred, dim=-1) / delta_t         # [B,N,T-1]
    speed_gt = torch.norm(dv_gt, dim=-1) / delta_t             # [B,N,T-1]


    heading_pred = prediction[..., 2]          # [B,N,T]
    heading_gt = ground_truth[..., 2]          # [B,N,T]

    dtheta_pred = wrap_to_pi_torch(heading_pred[:, :, 1:] - heading_pred[:, :, :-1])  # [B,N,T-1]
    dtheta_gt = wrap_to_pi_torch(heading_gt[:, :, 1:] - heading_gt[:, :, :-1])        # [B,N,T-1]

    valid_step = valid[:, :, 1:] & valid[:, :, :-1]          # [B,N,T-1]

    if valid_step.sum() == 0:
        return prediction.new_tensor(0.0)

    mask = valid_step

    speed_loss = F.smooth_l1_loss(speed_pred[mask], speed_gt[mask])
    heading_loss = F.smooth_l1_loss(dtheta_pred[mask], dtheta_gt[mask])

    return speed_loss + heading_loss



def collision_regularizer(prediction, ground_truth, plan, safe_distance=2.0):

    B, N, T, _ = prediction.shape


    nbr_pos = prediction[..., :2]                           # [B,N,T,2]
    valid_nbr = torch.ne(ground_truth, 0)[:, :, :, 0]       # [B,N,T]

    ego_pos = plan[..., :2]                                 # [B,T,2]
    valid_ego = torch.ne(plan, 0)[:, :, 0]                  # [B,T]

    ego_pos_exp = ego_pos.unsqueeze(1).expand(-1, N, -1, -1)    # [B,N,T,2]

    dist = torch.norm(nbr_pos - ego_pos_exp, dim=-1)        # [B,N,T]

    valid = valid_nbr & valid_ego.unsqueeze(1)              # [B,N,T]

    if valid.sum() == 0:
        return prediction.new_tensor(0.0)

    penalty = F.relu(safe_distance - dist)                  # [B,N,T]

    loss = (penalty * valid.float()).sum() / (valid.float().sum() + 1e-6)

    return loss


def calculate_metrics(prediction, ground_truth):
    valid = torch.ne(ground_truth, 0)[:, :, :, 0, None]
    prediction = prediction * valid

    prediction_error = torch.norm(prediction[:, :, :, :2] - ground_truth[:, :, :, :2], dim=-1)

    predictorADE = torch.mean(prediction_error, dim=-1)
    predictorADE = torch.masked_select(predictorADE, valid[:, :, 0, 0])
    predictorADE = torch.mean(predictorADE)

    predictorFDE = prediction_error[:, :, -1]
    predictorFDE = torch.masked_select(predictorFDE, valid[:, :, 0, 0])
    predictorFDE = torch.mean(predictorFDE)

    return predictorADE, predictorFDE


def transform_to_world_postion(traj, ego_current):
    o_x = ego_current[0]
    o_y = ego_current[1]
    o_theta = ego_current[2]
    x = traj[:, 0]
    y = traj[:, 1]

    world_x = x * torch.cos(o_theta) - y * torch.sin(o_theta) + o_x
    world_y = x * torch.sin(o_theta) + y * torch.cos(o_theta) + o_y
    traj = torch.stack([world_x, world_y], dim=-1)

    return traj