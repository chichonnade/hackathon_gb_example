"""
Demonstrates an object identification, grasping, and placement pipeline using a simulated robot.
"""

import re
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from client import GeneralBionixClient, PointCloudData, Grasp
from sim import (
    SimGrasp,
    ObjectInfo,
    CUBE_ORIENTATION,
    CUBE_SCALING,
    SPHERE_ORIENTATION,
    SPHERE_SCALING,
    TRAY_ORIENTATION,
    TRAY_SCALING,
    TRAY_POS,
    CUBE_RGBA,
    SPHERE_RGBA,
    SPHERE_MASS
)
from utils import compute_mask_center_of_mass, downsample_pcd, upsample_pcd
from visual_prompt.visual_prompt import VisualPrompterGrounding
from visual_prompt.utils import display_image
from vis_grasps import launch_visualizer, vis_grasps_meshcat
from transform import transform_pcd_cam_to_rob

# Path for the additional robot arm URDFs
SECOND_ARM_URDF = "piper_description/urdf/piper_description_virtual_eef_free_gripper.urdf"
THIRD_ARM_URDF = "piper_description/urdf/piper_description_virtual_eef_free_gripper.urdf"
FOURTH_ARM_URDF = "piper_description/urdf/piper_description_virtual_eef_free_gripper.urdf"

# GPT-4o prompt
USER_QUERY = "Identify the red cubes. If no red cubes are available then return an empty list."

# Configuration and simulation parameters
API_KEY = ""  # Use your API key here
URDF_PATH = "piper_description/urdf/piper_description_virtual_eef_free_gripper.urdf"
FREQUENCY = 30
CONFIG_PATH = 'visual_prompt/config/visual_prompt_config.yaml'

SIMULATION_OBJECTS = [
    ObjectInfo(
        urdf_path="cube_small.urdf",
        position=[0.3, 0.0, 0.025],
        orientation=CUBE_ORIENTATION,
        scaling=CUBE_SCALING,
        color=CUBE_RGBA
    ),
    ObjectInfo(
        urdf_path="cube_small.urdf",
        position=[0.23, 0.05, 0.025],
        orientation=CUBE_ORIENTATION,
        scaling=CUBE_SCALING,
        color=CUBE_RGBA
    ),
    ObjectInfo(
        urdf_path="tray/traybox.urdf",
        position=TRAY_POS,
        orientation=TRAY_ORIENTATION,
        scaling=TRAY_SCALING
    ),
    ObjectInfo(
        urdf_path="sphere2.urdf",
        position=[0.35, 0.1, 0.025],
        orientation=SPHERE_ORIENTATION,
        scaling=SPHERE_SCALING,
        color=SPHERE_RGBA,
        mass=SPHERE_MASS
    )
]

def find_empty_spot(grounder, image, marker_data, cube_pixel_size):
    """
    Ask the LLM to find a pixel coordinate where a cube of the given size can be placed
    without overlapping objects. Returns (x, y) pixel coordinates.
    """
    prompt = (
        f"Given the scene and segmentation masks, find a center pixel coordinate "
        f"where a {cube_pixel_size}×{cube_pixel_size} pixel square can be placed "
        "without overlapping existing objects. Return as two integers 'x,y'."
    )
    _, text_response, _ = grounder.request(text_query=prompt, image=image.copy(), data=marker_data)
    match = re.match(r"(\d+),\s*(\d+)", text_response.strip())
    if not match:
        raise ValueError(f"Unable to parse placement coordinates: '{text_response}'")
    return int(match.group(1)), int(match.group(2))

def main():
    # Initialize simulation environment, client, visualizer, and LLM grounder
    env = SimGrasp(urdf_path=URDF_PATH, frequency=FREQUENCY, objects=SIMULATION_OBJECTS)
    
    # Add additional robot arms as objects
    second_arm_id = env.add_object(
        urdf_path=SECOND_ARM_URDF,
        pos=[-0.3, 0.0, 0.025],
        orientation=CUBE_ORIENTATION,
        globalScaling=1.0
    )
    
    # Add third robot arm
    third_arm_id = env.add_object(
        urdf_path=THIRD_ARM_URDF,
        pos=[0.3, -0.3, 0.025],
        orientation=CUBE_ORIENTATION,
        globalScaling=1.0
    )
    
    # Add fourth robot arm
    fourth_arm_id = env.add_object(
        urdf_path=FOURTH_ARM_URDF,
        pos=[-0.3, -0.3, 0.025],
        orientation=CUBE_ORIENTATION,
        globalScaling=1.0
    )
    
    client = GeneralBionixClient(api_key=API_KEY)
    vis = launch_visualizer()
    grounder = VisualPrompterGrounding(CONFIG_PATH, debug=True)

    while True:
        # Step 1 & 2: Render and segment
        color, depth, _ = env.render_camera()
        pcd = env.create_pointcloud(color, depth)
        image, seg = env.obs['image'], env.obs['seg']
        obj_ids = np.unique(seg)[1:]
        all_masks = np.stack([seg == objID for objID in obj_ids])
        marker_data = {'masks': all_masks, 'labels': obj_ids}

        # Step 3: Visual prompt and object ID
        visual_prompt, _ = grounder.prepare_image_prompt(image.copy(), marker_data)
        display_image(visual_prompt[-1], (6,6))
        print("Calling GPT-4o to identify targets...")
        _, _, target_ids = grounder.request(text_query=USER_QUERY, image=image.copy(), data=marker_data)
        if len(target_ids) == 0:
            print("No target objects identified. Exiting.")
            break

        # Step 5: Select target and compute center
        selected_target_id = target_ids[-1]
        print(f"Selected target ID: {selected_target_id}")
        center_x, center_y = compute_mask_center_of_mass(
            marker_data['masks'][marker_data['labels'].tolist().index(selected_target_id)]
        )

        # Step 6: Crop point cloud
        pcd_ds = downsample_pcd(pcd, 4)
        cropped_data = client.crop_point_cloud(
            PointCloudData(points=np.array(pcd_ds.points).tolist(),
                           colors=np.array(pcd_ds.colors).tolist()),
            int(center_x/4), int(center_y)
        )
        cropped_pcd = o3d.geometry.PointCloud()
        cropped_pcd.points = o3d.utility.Vector3dVector(np.array(cropped_data.points))
        cropped_pcd.colors = o3d.utility.Vector3dVector(np.array(cropped_data.colors))
        cropped_pcd_full = upsample_pcd(cropped_pcd, pcd, 4)

        # Step 7 & 8: Transform and predict grasps
        cam_grasps = client.predict_grasps(
            PointCloudData(
                points=np.array(transform_pcd_cam_to_rob(cropped_pcd_full).points).tolist(),
                colors=np.array(transform_pcd_cam_to_rob(cropped_pcd_full).colors).tolist()
            )
        )
        robot_grasps = cam_grasps.grasps
        filtered = client.filter_grasps(robot_grasps)
        valid_idxs, valid_angles = filtered.valid_grasp_idxs, filtered.valid_grasp_joint_angles
        if not valid_idxs:
            print("No valid grasps. Restarting loop.")
            continue
        valid_grasps = [robot_grasps[i] for i in valid_idxs]

        # Step 9: Visualize and choose grasp
        vis_grasps_meshcat(vis, valid_grasps, transform_pcd_cam_to_rob(pcd))
        chosen_idx = 0
        chosen = valid_grasps[chosen_idx]
        print(f"Executing grasp at {chosen.translation}")
        env.add_debug_point(chosen.translation)

        # Step 10: Execute grasp
        env.grasp(valid_angles[chosen_idx])
        
        # We can't directly control the additional robot arms as they're just objects
        # Added as static fixtures in the simulation environment
        
        env.drop_object_in_tray()

        # Step 11: Place a cube in an empty spot
        obj_ids = np.unique(seg)[1:]
        all_masks = np.stack([seg == objID for objID in obj_ids])
        marker_data = {'masks': all_masks, 'labels': obj_ids}
        cube_px = int(CUBE_SCALING * image.shape[1])
        x_px, y_px = find_empty_spot(grounder, image, marker_data, cube_px)
        world_x, world_y, world_z = env.pixel_to_world(x_px, y_px, depth[int(y_px), int(x_px)])
        print(f"Placing cube at world coords: ({world_x:.3f}, {world_y:.3f}, {world_z:.3f})")
        env.add_object(
            urdf_path="cube_small.urdf",
            position=[world_x, world_y, world_z],
            orientation=CUBE_ORIENTATION,
            scaling=CUBE_SCALING,
            color=CUBE_RGBA
        )

if __name__ == "__main__":
    main()
