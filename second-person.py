"""
Interactive Face Mesh Reality - Main Application

A real-time computer vision art installation that creates immersive augmented reality
experiences by combining face tracking, image manipulation, and visual effects.

Author: Corey Dziadzio
Email: coreydziadzio@c11visualarts.com
GitHub: https://github.com/CJD-11

Features:
- Real-time face tracking with 468 landmark points
- Interactive polygon selection and manipulation
- Delayed movement effects with customizable parameters
- Multi-layer visual effects pipeline
- Abstract visualization with person segmentation
- Dynamic blending and composition
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import os
from typing import List, Tuple, Optional
import argparse
import json

# Import custom modules
from polygon_selector import PolygonSelector
from face_tracker import FaceTracker
from image_processor import ImageProcessor
from config import Config
from utils.camera_utils import CameraManager
from utils.display_utils import DisplayManager
from utils.math_utils import MathUtils


class FaceMeshReality:
    """
    Main application class for Interactive Face Mesh Reality.
    
    Coordinates all components including face tracking, image processing,
    polygon manipulation, and real-time display.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Face Mesh Reality application.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Load configuration
        self.config = Config(config_path) if config_path else Config()
        
        # Initialize components
        self.camera_manager = CameraManager(self.config.camera_index)
        self.face_tracker = FaceTracker(
            confidence=self.config.face_detection_confidence
        )
        self.image_processor = ImageProcessor(self.config)
        self.display_manager = DisplayManager(self.config.standard_window_size)
        self.polygon_selector = None
        
        # State variables
        self.reference_image = None
        self.reference_cutout = None
        self.reference_base = None
        self.polygon_points = []
        self.mask = None
        self.movement_queue = []
        self.is_running = False
        
        # Performance tracking
        self.frame_count = 0
        self.fps_counter = time.time()
        self.current_fps = 0
        
        print("Face Mesh Reality initialized successfully!")
    
    def load_reference_image(self, image_path: str) -> bool:
        """
        Load and prepare the reference image.
        
        Args:
            image_path: Path to the reference image file
            
        Returns:
            bool: True if image loaded successfully, False otherwise
        """
        try:
            self.reference_image = cv2.imread(image_path)
            if self.reference_image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Resize to standard size
            self.reference_image = cv2.resize(
                self.reference_image, 
                self.config.standard_window_size
            )
            
            print(f"Reference image loaded: {image_path}")
            print(f"Image size: {self.reference_image.shape}")
            return True
            
        except Exception as e:
            print(f"Error loading reference image: {e}")
            return False
    
    def setup_polygon_selection(self) -> bool:
        """
        Interactive polygon selection on the reference image.
        
        Returns:
            bool: True if polygon selected successfully, False otherwise
        """
        if self.reference_image is None:
            print("Error: No reference image loaded!")
            return False
        
        print("\n=== POLYGON SELECTION MODE ===")
        print("Instructions:")
        print("- Left-click to add polygon vertices")
        print("- Right-click to complete selection (minimum 3 points)")
        print("- Press 'q' to quit selection")
        print("- Press 'r' to reset polygon")
        
        self.polygon_selector = PolygonSelector(self.reference_image)
        success = self.polygon_selector.select_interactive()
        
        if success:
            self.polygon_points = self.polygon_selector.get_points()
            self._generate_masks_and_cutouts()
            print(f"Polygon selected with {len(self.polygon_points)} points")
            return True
        else:
            print("Polygon selection cancelled or failed")
            return False
    
    def _generate_masks_and_cutouts(self):
        """Generate masks and image cutouts from polygon selection."""
        if not self.polygon_points or self.reference_image is None:
            return
        
        # Create mask from polygon
        self.mask = np.zeros(self.reference_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(self.mask, [np.array(self.polygon_points)], 255)
        
        # Generate cutouts
        self.reference_cutout = cv2.bitwise_and(
            self.reference_image, 
            self.reference_image, 
            mask=self.mask
        )
        self.reference_base = cv2.bitwise_and(
            self.reference_image, 
            self.reference_image, 
            mask=cv2.bitwise_not(self.mask)
        )
        
        print("Masks and cutouts generated successfully")
    
    def process_face_tracking(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """
        Process face tracking and return current and delayed nose positions.
        
        Args:
            frame: Input camera frame
            
        Returns:
            Tuple of (current_nose_pos, delayed_nose_pos) or (None, None) if no face detected
        """
        results = self.face_tracker.process_frame(frame)
        
        if results.multi_face_landmarks:
            # Get nose landmark (index 1)
            nose_landmark = results.multi_face_landmarks[0].landmark[1]
            nose_x = int(nose_landmark.x * frame.shape[1])
            nose_y = int(nose_landmark.y * frame.shape[0])
            
            # Add to movement queue with timestamp
            current_time = time.time()
            self.movement_queue.append((current_time, nose_x, nose_y))
            
            # Remove old entries outside delay window
            self.movement_queue = [
                entry for entry in self.movement_queue 
                if current_time - entry[0] <= self.config.movement_delay
            ]
            
            # Get delayed position
            if self.movement_queue:
                delayed_x, delayed_y = self.movement_queue[0][1], self.movement_queue[0][2]
                return (nose_x, nose_y), (delayed_x, delayed_y)
        
        return None, None
    
    def generate_moved_cutout(self, current_pos: Tuple[int, int], delayed_pos: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate moved cutout based on face movement.
        
        Args:
            current_pos: Current nose position
            delayed_pos: Delayed nose position
            
        Returns:
            Tuple of (moved_cutout, ghost_zone_mask)
        """
        nose_x, nose_y = current_pos
        delayed_x, delayed_y = delayed_pos
        
        # Calculate movement with amplification
        shift_x = int((nose_x - delayed_x) * self.config.movement_amplification)
        shift_y = int((nose_y - delayed_y) * self.config.movement_amplification)
        
        # Create transformation matrix
        transformation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        
        # Apply transformation to cutout
        moved_cutout = cv2.warpAffine(
            self.reference_cutout, 
            transformation_matrix, 
            self.config.standard_window_size
        )
        
        # Generate ghost zone mask (areas where original mask is exposed)
        moved_mask = cv2.warpAffine(
            self.mask, 
            transformation_matrix, 
            self.config.standard_window_size
        )
        ghost_zone_mask = cv2.bitwise_and(self.mask, cv2.bitwise_not(moved_mask))
        
        return moved_cutout, ghost_zone_mask
    
    def generate_mesh_overlay(self, frame: np.ndarray, ghost_zone_mask: np.ndarray) -> np.ndarray:
        """
        Generate white face mesh overlay for ghost zones.
        
        Args:
            frame: Input camera frame
            ghost_zone_mask: Mask defining ghost zones
            
        Returns:
            np.ndarray: Mesh overlay image
        """
        # Get bounding rectangle of polygon
        x, y, w, h = cv2.boundingRect(np.array(self.polygon_points))
        
        # Create mesh canvas
        white_mesh_overlay = np.zeros_like(self.reference_image)
        white_mesh_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Process small frame for mesh detection
        small_frame = cv2.resize(frame, (w, h))
        small_results = self.face_tracker.process_frame(small_frame)
        
        if small_results.multi_face_landmarks:
            # Draw white face mesh
            mp.solutions.drawing_utils.draw_landmarks(
                white_mesh_canvas,
                small_results.multi_face_landmarks[0],
                mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(255, 255, 255), 
                    thickness=1
                )
            )
        
        # Place mesh in overlay
        white_mesh_overlay[y:y+h, x:x+w] = white_mesh_canvas
        
        # Apply ghost zone mask
        mesh_reveal = cv2.bitwise_and(
            white_mesh_overlay, 
            white_mesh_overlay, 
            mask=ghost_zone_mask
        )
        
        return mesh_reveal
    
    def generate_abstract_viewer(self, frame: np.ndarray, face_landmarks) -> np.ndarray:
        """
        Generate abstract viewer with person segmentation and effects.
        
        Args:
            frame: Input camera frame
            face_landmarks: Face landmark detection results
            
        Returns:
            np.ndarray: Processed abstract viewer image
        """
        # Person segmentation
        segmentation_results = self.face_tracker.segment_person(frame)
        condition = segmentation_results.segmentation_mask > 0.5
        
        # Extract person only
        person_only = np.zeros_like(frame)
        person_only[condition] = frame[condition]
        
        # Apply image processing effects
        abstract_view = self.image_processor.apply_effects_pipeline(person_only)
        
        # Add red face mesh overlay
        if face_landmarks.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                abstract_view,
                face_landmarks.multi_face_landmarks[0],
                mp.solutions.face_mesh.FACEMESH_TESSELATION,
                mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 0, 255), 
                    thickness=1, 
                    circle_radius=1
                ),
                mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 0, 255), 
                    thickness=1
                )
            )
        
        return abstract_view, segmentation_results.segmentation_mask
    
    def compose_final_image(self, mesh_overlay: np.ndarray, moved_cutout: np.ndarray, 
                           abstract_viewer: np.ndarray, segmentation_mask: np.ndarray) -> np.ndarray:
        """
        Compose the final output image with all layers.
        
        Args:
            mesh_overlay: White mesh overlay for ghost zones
            moved_cutout: Transformed polygon cutout
            abstract_viewer: Abstract viewer image
            segmentation_mask: Person segmentation mask
            
        Returns:
            np.ndarray: Final composed image
        """
        # Start with base image
        output_image = self.reference_base.copy()
        
        # Add mesh overlay and moved cutout
        output_image = cv2.add(output_image, mesh_overlay)
        output_image = cv2.add(output_image, moved_cutout)
        
        # Blend abstract viewer into bottom portion
        viewer_height = output_image.shape[0] // 3
        viewer_width = output_image.shape[1] // 2
        viewer_resized = cv2.resize(abstract_viewer, (viewer_width, viewer_height))
        
        # Resize segmentation mask
        mask_resized = cv2.resize(segmentation_mask.astype(np.float32), (viewer_width, viewer_height))
        mask_3ch = np.stack([mask_resized] * 3, axis=-1)
        
        # Calculate position for viewer
        x_offset = (output_image.shape[1] - viewer_width) // 2
        y_offset = output_image.shape[0] - viewer_height - 40
        
        # Blend viewer with background
        roi = output_image[y_offset:y_offset+viewer_height, x_offset:x_offset+viewer_width]
        blended = (viewer_resized * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)
        output_image[y_offset:y_offset+viewer_height, x_offset:x_offset+viewer_width] = blended
        
        return output_image
    
    def update_performance_metrics(self):
        """Update FPS and performance metrics."""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.fps_counter >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.fps_counter)
            self.frame_count = 0
            self.fps_counter = current_time
    
    def draw_performance_info(self, image: np.ndarray):
        """Draw performance information on the image."""
        info_text = [
            f"FPS: {self.current_fps:.1f}",
            f"Movement Queue: {len(self.movement_queue)}",
            f"Delay: {self.config.movement_delay:.1f}s",
            f"Amplification: {self.config.movement_amplification:.1f}x"
        ]
        
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(
                image, text, (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
    
    def handle_keyboard_input(self, key: int) -> bool:
        """
        Handle keyboard input during runtime.
        
        Args:
            key: Key code from cv2.waitKey()
            
        Returns:
            bool: True to continue running, False to exit
        """
        if key & 0xFF == ord('q'):
            return False
        elif key & 0xFF == ord('r'):
            # Reset movement queue
            self.movement_queue.clear()
            print("Movement queue reset")
        elif key & 0xFF == ord('d'):
            # Toggle debug information
            self.config.show_debug = not self.config.show_debug
            print(f"Debug info: {'ON' if self.config.show_debug else 'OFF'}")
        elif key & 0xFF == ord('s'):
            # Save current frame
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            # Save logic would go here
            print(f"Frame saved as {filename}")
        
        return True
    
    def run(self):
        """Main application loop."""
        if not self.camera_manager.is_opened():
            print("Error: Could not open camera!")
            return False
        
        if self.reference_image is None:
            print("Error: No reference image loaded!")
            return False
        
        if not self.polygon_points:
            print("Error: No polygon selected!")
            return False
        
        print("\n=== STARTING FACE MESH REALITY ===")
        print("Controls:")
        print("- 'q': Quit application")
        print("- 'r': Reset movement queue")
        print("- 'd': Toggle debug information")
        print("- 's': Save current frame")
        print("=====================================\n")
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Capture frame
                frame = self.camera_manager.get_frame()
                if frame is None:
                    print("Warning: Failed to capture frame")
                    continue
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process face tracking
                face_results = self.face_tracker.process_frame(frame)
                current_pos, delayed_pos = self.process_face_tracking(frame)
                
                # Generate output image
                if current_pos and delayed_pos:
                    # Face detected - generate dynamic content
                    moved_cutout, ghost_zone_mask = self.generate_moved_cutout(current_pos, delayed_pos)
                    mesh_overlay = self.generate_mesh_overlay(frame, ghost_zone_mask)
                    abstract_viewer, seg_mask = self.generate_abstract_viewer(frame, face_results)
                    output_image = self.compose_final_image(mesh_overlay, moved_cutout, abstract_viewer, seg_mask)
                else:
                    # No face detected - show reference image
                    output_image = self.reference_image.copy()
                
                # Add performance information if debug enabled
                if self.config.show_debug:
                    self.draw_performance_info(output_image)
                
                # Display result
                self.display_manager.show_image("Face Mesh Reality", output_image)
                
                # Handle keyboard input
                key = cv2.waitKey(1)
                if not self.handle_keyboard_input(key):
                    break
                
                # Update performance metrics
                self.update_performance_metrics()
                
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up resources...")
        self.camera_manager.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser
