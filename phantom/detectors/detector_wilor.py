"""
Wrapper around WiLoR for 3D hand pose estimation.

This script is designed as a drop-in replacement for 'detector_hamer.py' 
for the 'Phantom' project. It assumes that hand bounding boxes (`bboxes`) 
and handedness (`is_right`) are provided by an upstream detection pipeline,
as is the case in the Phantom framework.
"""
import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union

# ADD WILOR TO PYTHON PATH
wilor_path = os.path.join(os.path.dirname(__file__), '..', '..', 'submodules', 'WiLoR')
wilor_path = os.path.abspath(wilor_path)
if wilor_path not in sys.path:
    sys.path.insert(0, wilor_path)

import cv2
import torch
from wilor.utils import recursive_to  # CHANGED: wilor instead of hamer
from wilor.models import load_wilor  # CHANGED: WiLoR loading function
from wilor.datasets.vitdet_dataset import ViTDetDataset  # CHANGED: wilor dataset
from wilor.utils.renderer import cam_crop_to_full  # CHANGED: wilor utils
import matplotlib.pyplot as plt
from ultralytics import YOLO  # NEW: WiLoR uses YOLO for detection -- here not needed


from phantom.utils.data_utils import get_parent_folder_of_package

logger = logging.getLogger(__name__)

class DetectorWiLoR:
    """
    Detector using the WiLoR model for 3D hand pose estimation.
    
    This class matches the interface of `DetectorHamer` and is intended
    to be used within the Phantom framework. It loads the WiLoR reconstruction
    model and uses it to predict 3D hand pose from bounding boxes
    provided by an external detector.
    """
    def __init__(self, batch_size: int = 1, rescale_factor: float = 2.0):
        
        try:
           
            detector_file_path = Path(__file__)
            repo_root = detector_file_path.parent.parent.parent
            wilor_root_dir = repo_root / "submodules" / "WiLoR"

            checkpoint_path = wilor_root_dir / "pretrained_models" / "wilor_final.ckpt"
            cfg_path = wilor_root_dir / "pretrained_models" / "model_config.yaml"

            if not wilor_root_dir.exists():
                raise FileNotFoundError(f"Calculated WiLoR root directory not found at: {wilor_root_dir}. Please check this path.")
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"WiLoR checkpoint not found at: {checkpoint_path}")
            if not cfg_path.exists():
                raise FileNotFoundError(f"WiLoR config not found at: {cfg_path}")
                
        except Exception as e:
            logger.error(f"Error finding WiLoR models: {e}", exc_info=True)
            logger.error("Please ensure 'wilor' submodule is at '<phantom_root>/submodules/WiLoR/'")
            raise

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading WiLoR model on device: {self.device}")

        # --- Load WiLoR Model ---
        self.model, self.model_cfg = load_wilor(checkpoint_path=str(checkpoint_path), cfg_path=str(cfg_path))
        self.model.to(self.device)
        self.model.eval()
        logger.info("WiLoR model loaded successfully.")

        # --- Set MANO faces (for visualization) ---
        self.faces_right = self.model.mano.faces
        self.faces_left = self.faces_right[:,[0,2,1]]
        
        # --- Config from HaMeR script ---
        self.rescale_factor = rescale_factor
        self.batch_size = batch_size

    def detect_hand_keypoints(self, 
                              img: np.ndarray,
                              hand_side: Optional[str] = None,
                              visualize: bool=False, 
                              visualize_3d: bool=False, 
                              pause_visualization: bool=True, 
                              bboxes: Optional[np.ndarray]=None,
                              is_right: Optional[np.ndarray]=None,
                              kpts_2d_only: Optional[bool]=False,
                              camera_params: Optional[dict]=None) -> Optional[dict[str, any]]:
        """
        Detect hand keypoints in the input image using WiLoR.
        
        This method performs 3D reconstruction based on the provided
        bounding boxes (`bboxes`) and handedness (`is_right`).
        
        Args:
            img: Input RGB image as numpy array
            visualize: If True, displays 2D detection results in a window
            visualize_3d: If True, shows 3D visualization of keypoints and mesh
            pause_visualization: If True, waits for key press when visualizing
            bboxes: (N, 4) Bounding boxes of the hands (xyxy)
            is_right: (N,) Whether the hand is right (1 for right, 0 for left)
            kpts_2d_only: If True, use default focal length
            camera_params: Optional camera intrinsics (fx, fy, cx, cy)
            
        Returns:
            Dictionary containing 3D/2D keypoints, vertices, and camera info.
            Returns None if no bboxes are provided.
        """
        if bboxes is None or is_right is None or len(bboxes) == 0:
            logger.warning("No bounding boxes provided to DetectorWiLoR. Skipping.")
            return None

        # --- 1. Get Image/Camera Parameters ---
        # This logic is identical to hamer's, but will use self.model_cfg (from WiLoR)
        if not kpts_2d_only:
            scaled_focal_length, camera_center = self.get_image_params(img, camera_params)
        else:
            scaled_focal_length, camera_center = self.get_image_params(img, camera_params=None)

        # --- 2. Create Dataset/Dataloader ---
        # This uses the *exact* same ViTDetDataset class as HaMeR and WiLoR demos
        dataset = ViTDetDataset(self.model_cfg, img, bboxes, is_right, rescale_factor=self.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        list_2d_kpts, list_3d_kpts, list_verts = [], [], []
        T_cam_pred_all: list[torch.Tensor] = []
        list_global_orient = []
        list_hand_pose = []

        try:
            # --- 3. Run Inference Loop ---
            for batch in dataloader:
                batch = recursive_to(batch, self.device)
                with torch.no_grad():
                    out = self.model(batch)

                # --- 4. Get Full Camera Translation ---
                
                batch_T_cam_pred_all = self.get_all_T_cam_pred(batch, out, scaled_focal_length, self.device)

                # --- 5. Parse Batch Results ---
                batch_size_n = batch['img'].shape[0]
                for idx in range(batch_size_n):
                    kpts_3d = out["pred_keypoints_3d"][idx].detach().cpu().numpy()
                    verts = out["pred_vertices"][idx].detach().cpu().numpy()
                    
                    # Get WiLoR-style handedness (0 for left, 1 for right)
                    is_right_flag = batch["right"][idx].cpu().numpy()
                    
                    # Store MANO parameters
                    global_orient = out["pred_mano_params"]["global_orient"][idx].detach().cpu().numpy()
                    hand_pose = out["pred_mano_params"]["hand_pose"][idx].detach().cpu().numpy()
                    list_global_orient.append(global_orient)
                    list_hand_pose.append(hand_pose)

                    # --- WiLoR Handedness Flip (replaces hamer's static method) ---
                    # (2 * 1 - 1) = +1 (no change for right hand)
                    # (2 * 0 - 1) = -1 (flips x-axis for left hand)
                    flip = (2 * is_right_flag - 1)
                    verts[:, 0] = flip * verts[:, 0]
                    kpts_3d[:, 0] = flip * kpts_3d[:, 0]
                    
                    T_cam_pred = batch_T_cam_pred_all[idx].cpu().numpy()
                    img_w, img_h = batch["img_size"][idx].cpu().numpy()

                    # --- 6. Project to 2D ---
                    kpts_2d_wilor = self.project_full_img(kpts_3d, T_cam_pred, 
                                                          scaled_focal_length, camera_center, (img_w, img_h))

                    # Append results
                    list_2d_kpts.append(kpts_2d_wilor)
                    list_3d_kpts.append(kpts_3d + T_cam_pred) # Add translation for world coords
                    list_verts.append(verts + T_cam_pred)     # Add translation for world coords
                    T_cam_pred_all.append(batch_T_cam_pred_all[idx].cpu())
            
            if not list_2d_kpts:
                logger.warning("WiLoR model ran but produced no outputs.")
                return None

            # --- 7. Visualization (Copied from hamer) ---
            annotated_img = DetectorWiLoR.visualize_2d_kpt_on_img(
                kpts_2d=list_2d_kpts[0],
                img=img,
            )
            
            if visualize:
                cv2.imshow("Annotated Image (WiLoR)", annotated_img)
                cv2.waitKey(0 if pause_visualization else 1)

            if visualize_3d:
                DetectorWiLoR.visualize_keypoints_3d(annotated_img, list_3d_kpts[0], list_verts[0])

            # --- 8. Format Return Dictionary (Matches hamer) ---
            return {
                "annotated_img": annotated_img,
                "success": len(list_2d_kpts[0]) == 21,
                "kpts_3d": list_3d_kpts[0],
                "kpts_2d": np.rint(list_2d_kpts[0]).astype(np.int32),
                "verts": list_verts[0],
                "T_cam_pred": T_cam_pred_all[0],
                "scaled_focal_length": scaled_focal_length,
                "camera_center": camera_center.cpu().numpy() if isinstance(camera_center, torch.Tensor) else camera_center,
                "img_w": img.shape[1],
                "img_h": img.shape[0],
                "global_orient": list_global_orient[0],
                "hand_pose": list_hand_pose[0],
            }
        
        except Exception as e:
            logger.error(f"Error during WiLoR keypoint detection: {e}", exc_info=True)
            return None
    
    def get_image_params(self, img: np.ndarray, camera_params: Optional[dict]) -> Tuple[float, torch.Tensor]:
        """
        Get the scaled focal length and camera center.
        
        This version correctly uses the provided `camera_params` (fx, cx, cy)
        when available, falling back to the default (simplified) model
        settings only when `camera_params` is None.
        """
        img_w = img.shape[1]
        img_h = img.shape[0]
        
      
        if camera_params is not None:
            # Path 1: REAL intrinsics are provided
            scaled_focal_length = camera_params["fx"]
            
           
            camera_center = torch.tensor([[camera_params["cx"], camera_params["cy"]]], dtype=torch.float) 
            
            logger.debug(f"Using provided (REAL) camera_params: fx={scaled_focal_length}, c={camera_center}")
        
        else:
            
            scaled_focal_length = (self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE 
                                   * max(img_w, img_h))
            
            # Use the SIMPLIFIED image center
            camera_center = torch.tensor([[img_w / 2., img_h / 2.]], dtype=torch.float)
            
            logger.debug(f"Using default (SIMPLIFIED) focal length: {scaled_focal_length}")
       
            
        return scaled_focal_length, camera_center.to(self.device)
        
    @staticmethod
    def get_all_T_cam_pred(batch: dict, out: dict, scaled_focal_length: float, device: torch.device) -> torch.Tensor:
        """
        Get the camera transformation matrix in the full image frame.
        (Adapted from WiLoR's demo scripts)
        """
        multiplier = (2 * batch["right"] - 1).to(device)
        pred_cam = out["pred_cam"]
        
        # Flip y-translation for left hand
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        
        box_center = batch["box_center"].float().to(device)
        box_size = batch["box_size"].float().to(device)
        img_size = batch["img_size"].float().to(device) # W, H

        T_cam_pred_all = cam_crop_to_full(
            pred_cam, box_center, box_size, img_size, scaled_focal_length
        )
        return T_cam_pred_all
    
    @staticmethod
    def project_full_img(points_3d: Union[np.ndarray, torch.Tensor], 
                         cam_trans: Union[np.ndarray, torch.Tensor],
                         focal_length: float,
                         camera_center: Union[np.ndarray, torch.Tensor],  
                         img_res: Tuple[int, int]) -> np.ndarray:
        """
        Projects 3D keypoints to 2D image coordinates.
        Handles mixed Tensor/Numpy inputs robustly to fix TypeError.
        """
        # 1. Determine Device (prefer GPU if any input is already on GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(points_3d, torch.Tensor):
            device = points_3d.device
        elif isinstance(cam_trans, torch.Tensor):
            device = cam_trans.device

        # 2. Helper to safely convert anything to Tensor
        def to_tensor(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).float().to(device)
            elif isinstance(x, torch.Tensor):
                return x.float().to(device)
            # Handle lists or other iterables
            return torch.tensor(x).float().to(device)

        # 3. Convert all inputs
        points_3d_torch = to_tensor(points_3d)
        cam_trans_torch = to_tensor(cam_trans)
        cam_center_torch = to_tensor(camera_center).squeeze() # Ensure (2,) shape
        
        # 4. Construct K Matrix
        K = torch.eye(3, device=device)
        K[0, 0] = focal_length
        K[1, 1] = focal_length
        K[0, 2] = cam_center_torch[0]
        K[1, 2] = cam_center_torch[1]
        
        # 5. Project: Add translation (points + T)
        points_3d_trans = points_3d_torch + cam_trans_torch
        
        # Project (points @ K.T)
        projected_points = points_3d_trans @ K.T
        
        # Normalize to get 2D coords (x/z, y/z)
        points_2d = projected_points[..., :2] / (projected_points[..., 2:] + 1e-8)
        
        return np.rint(points_2d.cpu().numpy()).astype(np.int32)

    @staticmethod
    def project_3d_kpt_to_2d(kpts_3d: Union[np.ndarray, torch.Tensor], 
                             img_w: int, img_h: int, 
                             scaled_focal_length: float,
                             camera_center: Union[np.ndarray, torch.Tensor], 
                             T_cam: Optional[Union[np.ndarray, torch.Tensor]] = None) -> np.ndarray:
        """
        Project 3D keypoints to 2D image coordinates.
        Wrapper that delegates to project_full_img.
        """
        if T_cam is None:
            raise ValueError("T_cam must be provided for 2D projection")
            
        return DetectorWiLoR.project_full_img(
            points_3d=kpts_3d,
            cam_trans=T_cam,
            focal_length=scaled_focal_length,
            camera_center=camera_center,
            img_res=(img_w, img_h)
        )
    
    @staticmethod
    def visualize_keypoints_3d(annotated_img: np.ndarray, kpts_3d: np.ndarray, verts: np.ndarray) -> None:
        """
        3D visualization of keypoints and mesh.
        (Copied from detector_hamer.py)
        """
        nfingers = len(kpts_3d) - 1
        npts_per_finger = 4
        list_fingers = [np.vstack([kpts_3d[0], kpts_3d[i:i + npts_per_finger]]) for i in range(1, nfingers, npts_per_finger)]
        finger_colors_bgr = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 0, 255), (0, 255, 255)]
        finger_colors_rgb = [(color[2], color[1], color[0]) for color in finger_colors_bgr]
        
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(121, projection='3d')
        for finger_idx, finger_pts in enumerate(list_fingers):
            for i in range(len(finger_pts) - 1):
                color = finger_colors_rgb[finger_idx]
                ax1.plot(
                    [finger_pts[i][0], finger_pts[i + 1][0]],
                    [finger_pts[i][1], finger_pts[i + 1][1]],
                    [finger_pts[i][2], finger_pts[i + 1][2]],
                    color=np.array(color)/255.0,
                )
        ax1.scatter(kpts_3d[:, 0], kpts_3d[:, 1], kpts_3d[:, 2], s=10, c='black')
        # ax1.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=0.1) # Too slow
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        plt.title('3D Keypoints')

        ax2 = fig.add_subplot(122)
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        ax2.imshow(annotated_img_rgb)
        ax2.set_title('2D Projection')
        ax2.axis('off')

        plt.show()

    @staticmethod
    def visualize_2d_kpt_on_img(kpts_2d: np.ndarray, img: np.ndarray) -> np.ndarray:
        """
        Plot 2D hand keypoints on the image with finger connections.
        (Copied from detector_hamer.py)
        """
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 and img.shape[2] == 3 else img.copy()
        pts = kpts_2d.astype(np.int32)
        nfingers = len(pts) - 1
        npts_per_finger = 4
        list_fingers = [np.vstack([pts[0], pts[i:i + npts_per_finger]]) for i in range(1, nfingers, npts_per_finger)]
        finger_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 0, 255), (0, 255, 255)]
        thickness = 5 if img_bgr.shape[0] > 1000 else 2
        
        for finger_idx, finger_pts in enumerate(list_fingers):
            for i in range(len(finger_pts) - 1):
                color = finger_colors[finger_idx]
                cv2.line(
                    img_bgr,
                    tuple(finger_pts[i]),
                    tuple(finger_pts[i + 1]),
                    color,
                    thickness=thickness,
                )

        for pt in pts:
            cv2.circle(img_bgr, (pt[0], pt[1]), radius=thickness, color=(0,0,0), thickness=thickness-1)

        return img_bgr
        
    @staticmethod
    def annotate_bboxes_on_img(img: np.ndarray, debug_bboxes: dict) -> np.ndarray:
        """
        Annotate bounding boxes on the image.
        (This version is identical to the original detector_hamer.py)
        """
        color_dict = {
            "dino_bboxes": (0, 255, 0),
            "det_bboxes": (0, 0, 255),
            "refined_bboxes": (255, 0, 0),
            "filtered_bboxes": (255, 255, 0),
        }
        corner_dict = {
            "dino_bboxes": "top_left",
            "det_bboxes": "top_right",
            "refined_bboxes": "bottom_left",
            "filtered_bboxes": "bottom_right",
        }
        
        def draw_bbox_and_label(bbox, label, color, label_pos, include_label=True):
            """ Helper function to draw the bounding box and add label """
            # It implicitly uses 'img' from the outer scope
            cv2.rectangle(
                img,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color,
                2,
            )
            if include_label:
                # Reverted font size to 1 to match original
                cv2.putText(
                    img, label, label_pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA
                )

        
        label_pos_dict = {
            "top_left": lambda bbox: (int(bbox[0]), int(bbox[1]) - 10),
            "bottom_right": lambda bbox: (int(bbox[2]) - 150, int(bbox[3]) - 10),
            "top_right": lambda bbox: (int(bbox[2]) - 150, int(bbox[1]) - 10),
            "bottom_left": lambda bbox: (int(bbox[0]), int(bbox[3]) - 10),
        }

        # No longer creating 'annotated_img.copy()'
        for key, value in debug_bboxes.items():
            # Unpack bboxes and scores
            if key in ["dino_bboxes", "det_bboxes"]:
                bboxes, scores = value
            else:
                bboxes = value
                scores = [None] * len(bboxes)  

            color = color_dict.get(key, (0, 0, 0)) 
            label_pos_fn = label_pos_dict[corner_dict.get(key, "top_left")]

            # Draw each bounding box and its label
            for idx, bbox in enumerate(bboxes):
                score_text = f" {scores[idx]:.3f}" if scores[idx] is not None else ""
                label = key.split("_")[0] + score_text

                # Draw bounding box and label on the image
                label_pos = label_pos_fn(bbox)
                if key in ["dino_bboxes", "det_bboxes"] or idx == 0:
                    # This call now matches the helper function
                    draw_bbox_and_label(bbox, label, color, label_pos)
        
        # Returns the modified 'img' directly
        return img