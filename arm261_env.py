import gym
import mujoco
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt

class Arm261Env(gym.Env):
    def __init__(self):
        super(Arm261Env, self).__init__()
        #load the robotic arm
        self.model = mujoco.MjModel.from_xml_path("arm261.xml")
        self.data = mujoco.MjData(self.model)

        #define action spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        obs_size = self.model.nq + self.model.nv

        #define observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        #episode and step management
        self.episode_count = 0
        self.target_pos = np.zeros(3)  # target position
        self.max_steps = 500 #max number of steps per episode
        self.current_step = 0

        #MJT Logging
        self.mjt_data = []  # Store MJT comparison data for all episodes
        self.positions = []
        self.timestamps = []
        # Store positions and timestamps for all episodes
        self.all_episode_positions = {}  # Dictionary to store positions per episode
        self.all_episode_timestamps = {}  # Dictionary to store timestamps per episode


    def step(self, action):
        """Performs one step in the MuJoCo simulation."""

        #1. Clip and apply action to Mujoco simulation
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        # 2. Check if finger touched the target
        touched = self._check_finger_touch()
        if touched:
            print(f"✅ Episode {self.episode_count}: Finger touched the target box!")
            done = True  # End episode immediately
            reward = 1.0  # Give a positive reward for success
            obs = self._get_obs()
            self.episode_count += 1
            self.reset()
            return obs, reward, done, {}

        #3. Get observation
        obs = self._get_obs()
        self.current_step += 1

        #4. Retrieve Finger Tip position
        try:
            finger_tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "finger_tip")
            end_effector_pos = self.data.site_xpos[finger_tip_id].copy()
        except mujoco.MujocoException:
            print("🚨 Error: Finger tip site not found!")
            end_effector_pos = np.zeros(3)  # Default to avoid crashes

        # 5. Store Movement Data for MJT Analysis
        if not hasattr(self, 'all_episode_positions'):
            self.all_episode_positions = {}
            self.all_episode_timestamps = {}

        # Store position and timestamp for MJT analysis
        self.positions.append(end_effector_pos)
        self.timestamps.append(self.current_step * 0.01)  # Assuming 100Hz control rate

        # 6. Store episode trajectory for final analysis
        if self.episode_count not in self.all_episode_positions:
            self.all_episode_positions[self.episode_count] = []
            self.all_episode_timestamps[self.episode_count] = []

        self.all_episode_positions[self.episode_count].append(end_effector_pos)
        self.all_episode_timestamps[self.episode_count].append(self.current_step * 0.01)

        # 7. Compute distance to target
        distance = np.linalg.norm(self.target_pos - end_effector_pos)
        done = distance < 0.005 or self.current_step >= self.max_steps

        #8. Compute jerk cost if enough positions are available
        jerk_cost = self.compute_jerk() if len(self.positions) > 3 else 0  

        #9. Reward Function: Encourages reaching the target with minimal jerk
        reward = -distance + 0.01 * np.sum(np.abs(action)) - 0.1 * jerk_cost

        #10. End of episode handling
        if done:
            print(f"📊 Episode {self.episode_count} completed. Reward: {reward:.2f}, Avg Jerk: {jerk_cost:.5f}")
            self.episode_count += 1
            self.reset()

        return obs, reward, done, {}

    '''def step(self, action):
        """Performs one step in the MuJoCo simulation."""
        
        #1. Clip and apply action to Mujoco simulation
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        # 2.Check if finger touched the target
        touched = self._check_finger_touch()
        if touched:
            print(f"✅ Episode {self.episode_count}: Finger touched the target box!")

        #3. Get observation
        obs = self._get_obs()
        self.current_step += 1

        #4. Retrieve Finger Tip position
        try:
            finger_tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "finger_tip")
            end_effector_pos = self.data.site_xpos[finger_tip_id].copy()
        except mujoco.MujocoException:
            print("🚨 Error: Finger tip site not found!")
            end_effector_pos = np.zeros(3)  # Default to avoid crashes

        # 5. Store Movement Data for MJT Analysis
        if not hasattr(self, 'all_episode_positions'):
            self.all_episode_positions = {}
            self.all_episode_timestamps = {}

        # Store position and timestamp for MJT analysis
        self.positions.append(end_effector_pos)
        self.timestamps.append(self.current_step * 0.01)  # Assuming 100Hz control rate

        # 6. Store episode trajectory for final analysis
        if self.episode_count not in self.all_episode_positions:
            self.all_episode_positions[self.episode_count] = []
            self.all_episode_timestamps[self.episode_count] = []

        self.all_episode_positions[self.episode_count].append(end_effector_pos)
        self.all_episode_timestamps[self.episode_count].append(self.current_step * 0.01)

        # 7.  Compute distance to target (end-effector position and target position)
        distance = np.linalg.norm(self.target_pos - end_effector_pos)
        done = distance < 0.005 or self.current_step >= self.max_steps

        #8. Compute jerk cost if enough positions are available
        jerk_cost = self.compute_jerk() if len(self.positions) > 3 else 0  

        #9. Reward Function: Encourages reaching the target with minimal jerk
        reward = -distance + 0.01 * np.sum(np.abs(action)) - 0.1 * jerk_cost
        Encourages the agent to minimize distance to the target (-distance).
            Adds a small reward for taking action (+ 0.01 * action magnitude).
            Penalizes high jerk values (- 0.1 * jerk_cost) to encourage smooth movement.

        #10. End of episode handling
        if done:
            print(f"📊 Episode {self.episode_count} completed. Reward: {reward:.2f}, Avg Jerk: {jerk_cost:.5f}")
            self.episode_count += 1
            self.reset()

        return obs, reward, done, {}'''

    
    def _check_finger_touch(self):
        """Check if the arm's finger is in contact with the target box."""
        #retrieve object IDs
        finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "finger_geom")
        box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_box")

        #Compute Distance Between Finger and Box
        finger_pos = self.data.geom_xpos[finger_id]
        box_pos = self.data.geom_xpos[box_id]
        distance = np.linalg.norm(finger_pos - box_pos)
        #3. Check for Close Contact (Proximity Threshold)
        if distance < 0.01:  # Adjusst threshold
            print(f"⚠️ Close Contact Detected at Distance: {distance}")
            return True
        #4. Handle Missing Objects
        if finger_id == -1 or box_id == -1:
            print("Error: One of the geoms ('finger_geom' or 'target_box') was not found!")
            return False  # Return false to avoid crashing

        # 5. Check if these two are in contact
        for contact in self.data.contact:
            if (contact.geom1 == finger_id and contact.geom2 == box_id) or (contact.geom1 == box_id and contact.geom2 == finger_id):
                return True  # Contact detected

        return False  # No contact



    def compute_jerk(self):
        """Compute jerk (rate of change of acceleration) for the robotic arm."""
        if len(self.positions) < 4:
            return 0  

        positions = np.array(self.positions)
        timestamps = np.array(self.timestamps)

        # Compute velocity
        velocities = np.diff(positions, axis=0) / np.diff(timestamps)[:, None]

        # Compute acceleration
        accelerations = np.diff(velocities, axis=0) / np.diff(timestamps[:-1])[:, None]

        # Compute jerk (rate of change of acceleration)
        jerks = np.diff(accelerations, axis=0) / np.diff(timestamps[:-2])[:, None]

        # Compute jerk cost (sum of squared jerk norms)
        jerk_cost = np.sum(np.linalg.norm(jerks, axis=1) ** 2 * np.diff(timestamps[:-2]))

        return jerk_cost  

    def plot_jerk(self):
        """Plot jerk profile after an episode."""
        if len(self.positions) < 4:
            print("Not enough data to plot jerk.")
            return

        positions = np.array(self.positions)
        timestamps = np.array(self.timestamps)

        # Compute velocity
        velocities = np.diff(positions, axis=0) / np.diff(timestamps)[:, None]

        # Compute acceleration
        accelerations = np.diff(velocities, axis=0) / np.diff(timestamps[:-1])[:, None]

        # Compute jerk (rate of change of acceleration)
        jerks = np.diff(accelerations, axis=0) / np.diff(timestamps[:-2])[:, None]

        # Plot jerk profile
        plt.figure(figsize=(8, 5))
        plt.plot(timestamps[:-3], np.linalg.norm(jerks, axis=1), label="Jerk Magnitude")
        plt.xlabel("Time (s)")
        plt.ylabel("Jerk (m/s³)")
        plt.title("Jerk Profile of Robotic Arm")
        plt.legend()
        plt.show()

    
    def get_end_effector_position(self):
        """
        Returns the (x, y, z) position of the robot's end-effector.
        """
        end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        if end_effector_id == -1:
            raise ValueError("End-effector site not found in MuJoCo model. Check XML file.")

        return self.data.site_xpos[end_effector_id]  # Returns (x, y, z) position

    def get_target_position(self):
        """
        Returns the (x, y, z) position of the target box.
        """
        box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        if box_id == -1:
            raise ValueError("Box body not found in MuJoCo model. Check XML file.")

        return self.data.xpos[box_id]  # Returns (x, y, z) position

    def reset(self):
        #1. Reset the MuJoCo Environment
        mujoco.mj_resetData(self.model, self.data)  # Reset the environment
        self.current_step = 0

        #2.  Set fixed target position
        self.target_pos = np.array([-0.3,0.7, 0.01])  # Example: Fixed X, Y, Z
        # 3. Get body ID for the box
        box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        if box_body_id == -1:
            raise ValueError("Box body not found in MuJoCo model. Check XML file.")
        #4. Apply Changes to MuJoCo
        mujoco.mj_forward(self.model, self.data)  # Apply changes
        return self._get_obs()





    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    
    