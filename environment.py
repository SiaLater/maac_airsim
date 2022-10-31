import numpy as np
import airsim
import cv2
import copy
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import logging
import sys

logging.basicConfig(filename='experiment.log')


class AirSimEnv:
    def __init__(self, n_agents, save_obs=False):
        self.n_agents = n_agents
        self.save_obs = save_obs

        self.action_space = 2
        self.observation_space = 3

        self.drone_names = [f"Drone{i + 1}" for i in range(self.n_agents)]
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        for i in range(self.n_agents):
            self.client.enableApiControl(True, vehicle_name=self.drone_names[i])
            self.client.armDisarm(True, vehicle_name=self.drone_names[i])
        self.join_actions(self.client.takeoffAsync)
        self.join_actions(self.client.hoverAsync)
        self.positions = []
        self.origin_coords = []
        for i in range(self.n_agents):
            position = self.get_position(i)
            self.origin_coords.append(self.get_position(i))
            self.positions.append(position)

        self.all_victim_names = [c for c in self.client.simListSceneObjects() if 'rp' in c]
        self.n_targets = len(self.all_victim_names)
        xs, ys = [], []
        # decide boundary
        # for victim_name in self.all_victim_names:
        #     pos = self.client.simGetObjectPose(victim_name).position
        #     xs.append(pos.x_val)
        #     ys.append(pos.y_val)
        # print(xs)
        # xs = np.asarray(xs)
        # ys = np.asarray(ys)
        # self.x_max = np.max(xs)
        # self.x_min = np.min(xs)
        # self.y_max = np.max(ys)
        # self.y_min = np.min(ys)
        # self.altitude = -15
        # print(f"Boundaries: x: [{self.x_min}, {self.x_max}], y: [{self.y_min}, {self.y_max}]")

        self.steps = None
        self.camera_name = "3"
        self.image_type = airsim.ImageType.Scene
        for i in range(self.n_agents):
            self.client.enableApiControl(True, vehicle_name=self.drone_names[i])
            self.client.armDisarm(True, vehicle_name=self.drone_names[i])
            self.client.simSetDetectionFilterRadius(self.camera_name, self.image_type,
                                                    200 * 100, vehicle_name=self.drone_names[i])
            self.client.simAddDetectionFilterMeshName(self.camera_name, self.image_type,
                                                      "rp*", vehicle_name=self.drone_names[i])
        self.detected = set()
        self.detections = [None for _ in range(self.n_agents)]
        self.collisions = [False for _ in range(self.n_agents)]
        self.out_of_bounds = [False for _ in range(self.n_agents)]

        self.log = open("./log.txt", 'a')


    def detect(self):
        # detections = self.client.simListSceneObjects()
        detections = []
        for i in range(self.n_agents):
            drone_name = self.drone_names[i]
            rawImage = self.client.simGetImage(self.camera_name, self.image_type, vehicle_name=drone_name)
            detection = self.client.simGetDetections(self.camera_name, self.image_type, vehicle_name=drone_name)
            detections.append(set([d.name for d in detection]))
        return detections

    def plot_bbox(self, img, detections):
        for detection in detections:
            cv2.rectangle(img, (int(detection.box2D.min.x_val), int(detection.box2D.min.y_val)),
                          (int(detection.box2D.max.x_val), int(detection.box2D.max.y_val)), (255, 0, 0), 2)
            cv2.putText(img, detection.name, (int(detection.box2D.min.x_val), int(detection.box2D.min.y_val - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12))
        return img

    def write_log(self, txt):
        self.log.write(txt + '\n')

    def observation(self):
        obs = []
        for i in range(self.n_agents):
            drone_name = self.drone_names[i]
            self.client.hoverAsync(vehicle_name=drone_name).join()
            self.positions[i] = self.get_position(i)
            raw_img, = self.client.simGetImages([airsim.ImageRequest(self.camera_name, self.image_type, False, False)],
                                                vehicle_name=drone_name)
            # img_rgb = np.fromstring(raw_img, np.int8)
            img_rgb = np.frombuffer(raw_img.image_data_uint8, dtype=np.uint8).reshape(raw_img.height, raw_img.width, 3)

            # img_rgb = decode_img.reshape([raw_img.height, raw_img.width, 3])
            # print(img_rgb.shape)
            # img = cv2.imdecode(decode_img, cv2.IMREAD_UNCHANGED)
            # img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            img_bbox = copy.deepcopy(img_rgb)
            obs.append(([self.positions[i]], np.transpose(img_rgb, (2, 0, 1))))

            detection = self.client.simGetDetections(self.camera_name, self.image_type, vehicle_name=drone_name)
            self.detections[i] = set([d.name for d in detection])

            # if self.save_obs:
            #     img_bbox = self.plot_bbox(img_bbox, detection)
            #     cv2.imwrite(f"./pics/agent_{i}_{self.steps}.png", img_bbox)
        return obs

    def join_actions(self, action_func, kargs_list=None):
        fs = []
        for i in range(self.n_agents):
            self.client.armDisarm(True, vehicle_name=self.drone_names[i])
            if kargs_list:
                f = action_func(vehicle_name=self.drone_names[i], **kargs_list[i])
            else:
                f = action_func(vehicle_name=self.drone_names[i])
            fs.append(f)
        [f.join() for f in fs]

    def reset(self):
        self.write_log("New episode")
        self.client.reset()
        for i in range(self.n_agents):
            self.client.enableApiControl(True, vehicle_name=self.drone_names[i])
            self.client.armDisarm(True, vehicle_name=self.drone_names[i])
        self.join_actions(self.client.takeoffAsync)

        self.positions = []
        for i in range(self.n_agents):
            position = self.get_position(i)
            self.positions.append(position)
        self.steps = 0
        self.detected = set()
        self.detections = [-1 for _ in range(self.n_agents)]
        self.collisions = [False for _ in range(self.n_agents)]
        self.out_of_bounds = [False for _ in range(self.n_agents)]
        return self.observation()

    def get_position(self, idx):
        position = self.client.simGetVehiclePose(vehicle_name=self.drone_names[idx]).position
        return [position.x_val, position.y_val, position.z_val]

    def check_collision(self, dist_threshold=1):
        positions = [self.get_position(i) for i in range(self.n_agents)]
        for i in range(self.n_agents):
            if self.client.simGetCollisionInfo(vehicle_name=self.drone_names[i]).has_collided:
                self.collisions[i] = True
            else:
                self.collisions[i] = False

    def check_boundary(self):
        positions = [self.get_position(i) for i in range(self.n_agents)]
        for i in range(self.n_agents):
            x, y = positions[i][0], positions[i][1]
            if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
                self.out_of_bounds[i] = True
            else:
                self.out_of_bounds[i] = False

    def reward(self):
        rewards = np.zeros(self.n_agents)
        for i, detection in enumerate(self.detections):
            if len(detection) > 0:
                duplicate = len(detection & self.detected)
                if duplicate > 1:
                    rewards[i] -= 5
                else:
                    rewards[i] += 20
        self.detected = self.detected.union(set.union(*self.detections))
        for i in range(self.n_agents):
            if self.collisions[i]:
                rewards[i] = rewards[i] - 100
            if self.out_of_bounds[i]:
                rewards[i] = rewards[i] - 100
        if len(self.detected) == self.n_targets:
            rewards = rewards + 100
        return rewards

    def done(self):
        if len(self.detected) == self.n_targets or self.steps >= 50:
            return [True for _ in range(self.n_agents)]
        return [c or o for c, o in zip(self.collisions, self.out_of_bounds)]

    def step(self, actions):
        # print(self.detected)
        # for i in range(self.n_agents):
        #     vx, vy = actions[i]
        #     self.client.moveByVelocityZAsync(vx, vy, self.positions[i][2], 2, vehicle_name=self.drone_names[i]).join()
        kwargs_list = [{"vx": actions[i][0], "vy": actions[i][1], "z": self.altitude, "duration": 2}
                       for i in range(self.n_agents)]
        self.join_actions(self.client.moveByVelocityZAsync, kwargs_list)
        state_next = self.observation()
        self.check_collision()
        self.check_boundary()
        rewards = self.reward()
        dones = self.done()
        self.steps += 1
        return state_next, rewards, dones

    def __del__(self):
        self.log.close()
