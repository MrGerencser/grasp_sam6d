#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from franka_msgs.action import Grasp, Move
from sensor_msgs.msg import JointState
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
import time
from scipy.spatial.transform import Rotation as R_simple
from enum import Enum
import copy


class GraspState(Enum):
    """State machine states for grasp execution"""
    IDLE = "idle"
    MOVING_TO_HOME = "moving_to_home"
    OPENING_GRIPPER = "opening_gripper"
    MOVING_TO_PRE_GRASP = "moving_to_pre_grasp"
    MOVING_TO_GRASP = "moving_to_grasp"
    CLOSING_GRIPPER = "closing_gripper"
    CHECKING_GRASP = "checking_grasp"
    LIFTING_OBJECT = "lifting_object"
    RETURNING_TO_HOME = "returning_to_home"
    MOVING_TO_DROP_SAFE = "moving_to_drop_safe"
    MOVING_TO_DROP = "moving_to_drop"
    RELEASING_OBJECT = "releasing_object"
    RETREATING_FROM_DROP = "retreating_from_drop"
    FINAL_HOME = "final_home"
    ERROR_RECOVERY = "error_recovery"


class GraspFailureException(Exception):
    """Custom exception for grasp failures"""
    pass


class GraspExecutor(Node):
    def __init__(self):
        super().__init__('grasp_executor')
        
        # Configuration constants
        self.config = {
            'workspace_limits': {
                'x_min': 0.1, 'x_max': 0.8,
                'y_min': -0.5, 'y_max': 0.5,
                'z_min': 0.0, 'z_max': 0.8
            },
            'gripper': {
                'max_width': 0.08,
                'goal_width': 0.00,
                'speed': 0.05,
                'force': 50.0,
                'epsilon_inner': 0.05,
                'epsilon_outer': 0.07
            },
            'offsets': {
                'pre_grasp': 0.15,
                'approach': 0.05,
                'lift': 0.1,
                'drop_approach': 0.1,
                'drop_safe': 0.15
            },
            'timeouts': {
                'movement': 10.0,  # seconds to wait for movements
                'gripper': 5.0,    # seconds to wait for gripper actions
                'state_check': 0.1  # seconds between state checks
            },
            'drop_box': {
                'x': 0.2, 'y': 0.6, 'z': 0.18
            }
        }
        
        self.callback_group = ReentrantCallbackGroup()
        
        # State machine variables
        self.current_state = GraspState.IDLE
        self.pending_grasp_pose = None
        self.active_grasp_pose = None
        self.gripper_width = None
        self.state_start_time = None
        self.current_future = None
        self.grasp_successful = False

        # Robot pose tracking
        self.current_robot_pose = None
        self.target_pose = None
        self.pose_tolerance = {
            'position': 0.01,  # 10mm tolerance
            'orientation': 0.05  # ~3 degrees tolerance
        }
        
        # Setup communication and gripper
        self._setup_communication()
        self._setup_gripper()
        
        # Create reference poses
        self.home_pose = self._create_home_pose()
        self.drop_pose = self._create_drop_pose()
        
        # Create state machine timer
        self.state_timer = self.create_timer(
            self.config['timeouts']['state_check'],
            self._state_machine_callback,
            callback_group=self.callback_group
        )
        
        self.get_logger().info("Grasp Executor ready! Waiting for grasp poses...")
    
    def _setup_communication(self):
        """Setup ROS2 publishers and subscribers"""
        # Grasp pose subscriber
        # self.pose_subscriber = self.create_subscription(
        #     PoseStamped, '/perception/object_pose',
        #     self._on_grasp_pose_received, 1,
        #     callback_group=self.callback_group
        # )
        
        self.pose_subscriber = self.create_subscription(
            PoseStamped, '/sam6d/pose',
            self._on_grasp_pose_received, 1,
            callback_group=self.callback_group
        )
        

        # Robot end-effector pose subscriber
        self.current_pose_subscriber = self.create_subscription(
            PoseStamped, '/franka_robot_state_broadcaster/current_pose',
            self._on_current_pose_received,
            QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            ),
            callback_group=self.callback_group
        )
        
        # Robot Gripper width Subscriber
        self.joint_state_subscriber = self.create_subscription(
            JointState, '/fr3_gripper/joint_states',
            self._on_joint_state_received,
            QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            ),
            callback_group=self.callback_group
        )

        # Publishers
        self.cartesian_target_publisher = self.create_publisher(
            PoseStamped, '/cartesian_target_pose', 1
        )

        self.riemann_target_publisher = self.create_publisher(
            PoseStamped, '/riemannian_motion_policy/reference_pose', 1
        )
        
        self.status_publisher = self.create_publisher(
            Bool, '/robot/grasp_status', 1
        )
        
        self.execution_state_publisher = self.create_publisher(
            Bool, '/robot/grasp_executing', 1
        )
        
    def _on_current_pose_received(self, msg):
        """Update current robot end-effector pose"""
        self.current_robot_pose = msg
        
    def _movement_completed(self, timeout_seconds):
        """
        Check if movement to the target pose is completed or failed.

        Returns:
            'completed' if the robot is within tolerance of the target pose,
            'failed' if timeout is exceeded or pose data is missing,
            'in_progress' otherwise.
        """
        if self.target_pose is None or self.current_robot_pose is None:
            return 'failed'
        if self._poses_are_close(
            self.current_robot_pose, 
            self.target_pose,
            position_tolerance=0.04,
            orientation_tolerance=0.2
        ):
            return 'completed'
        if self._check_timeout(timeout_seconds):
            self.get_logger().warn(f"Movement to target timed out after {timeout_seconds}s")
            return 'failed'
        return 'in_progress'

    def _poses_are_close(self, pose1, pose2, position_tolerance=None, orientation_tolerance=None, timeout=0.1):
        """Check if two poses are within tolerance"""
        if pose1 is None or pose2 is None:
            return False

        pos_tol = position_tolerance or self.pose_tolerance['position']
        # ori_tol = orientation_tolerance or self.pose_tolerance['orientation']

        # Extract poses (handle both Pose and PoseStamped)
        p1 = pose1.pose if hasattr(pose1, 'pose') else pose1
        p2 = pose2.pose if hasattr(pose2, 'pose') else pose2

        # Position difference
        pos_diff = np.sqrt(
            (p1.position.x - p2.position.x)**2 +
            (p1.position.y - p2.position.y)**2 +
            (p1.position.z - p2.position.z)**2
        )
        
        if pos_diff > pos_tol:
            return False
        
        return True  # Orientation check is skipped for now
        
        # # Orientation difference (quaternion distance)
        # q1 = np.array([p1.orientation.x, p1.orientation.y, p1.orientation.z, p1.orientation.w])
        # q2 = np.array([p2.orientation.x, p2.orientation.y, p2.orientation.z, p2.orientation.w])
        
        # # Normalize quaternions
        # q1 = q1 / np.linalg.norm(q1)
        # q2 = q2 / np.linalg.norm(q2)
        
        # # Quaternion distance (angle between orientations)
        # dot_product = np.abs(np.dot(q1, q2))
        # dot_product = np.clip(dot_product, 0.0, 1.0)  # Numerical safety
        # angle_diff = 2 * np.arccos(dot_product)
        
        # return angle_diff <= ori_tol
    
    def _get_position_error(self):
        """Get current position error magnitude"""
        if self.current_robot_pose is None or self.target_pose is None:
            return float('inf')
        
        p1 = self.current_robot_pose.pose.position
        p2 = self.target_pose.pose.position
        
        return np.sqrt(
            (p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2
        )

    def _on_joint_state_received(self, msg):
        """Update gripper width from joint states"""
        try:
            idx1 = msg.name.index('fr3_finger_joint1')
            idx2 = msg.name.index('fr3_finger_joint2')
            self.gripper_width = msg.position[idx1] + msg.position[idx2]
        except (ValueError, IndexError):
            pass  # Joint names not found
    
    def _setup_gripper(self):
        """Setup gripper action clients"""
        self.move_client = ActionClient(
            self, Move, '/fr3_gripper/move', 
            callback_group=self.callback_group
        )
        self.grasp_client = ActionClient(
            self, Grasp, '/fr3_gripper/grasp', 
            callback_group=self.callback_group
        )
        
        # Wait for servers and home gripper
        self._wait_for_gripper_servers()
        self._home_gripper()
    
    def _wait_for_gripper_servers(self):
        """Wait for all gripper action servers"""
        servers = [
            (self.move_client, 'Move'),
            (self.grasp_client, 'Grasp')
        ]
        
        for client, name in servers:
            self.get_logger().info(f'Waiting for {name} action server...')
            while not client.wait_for_server(timeout_sec=2.0) and rclpy.ok():
                self.get_logger().info(f'{name} not available, waiting...')
            
            if rclpy.ok():
                self.get_logger().info(f'{name} action server found.')
            else:
                raise SystemExit('ROS shutdown while waiting for servers')
    
    def _home_gripper(self):
        """Home the gripper by opening it to max width"""
        self.get_logger().info(f"Homing gripper by opening to {self.config['gripper']['max_width']*1000:.0f}mm")
        goal = Move.Goal()
        goal.width = self.config['gripper']['max_width']
        goal.speed = self.config['gripper']['speed']
        self.move_client.send_goal_async(goal)
    
    def _create_home_pose(self):
        """Create home pose"""
        home_rotation = R_simple.from_euler('xyz', [np.pi, 0.0, 0.0])
        home_quat = home_rotation.as_quat()  # x, y, z, w
        return self._create_pose(
            0.5, 0.0, 0.35, 
            home_quat[3], home_quat[0], home_quat[1], home_quat[2]
        )
    
    def _create_drop_pose(self):
        """Create drop pose at the box location"""
        home_rotation = R_simple.from_euler('xyz', [np.pi, 0.0, 0.0])
        drop_quat = home_rotation.as_quat()
        
        return self._create_pose(
            self.config['drop_box']['x'],
            self.config['drop_box']['y'], 
            self.config['drop_box']['z'],
            drop_quat[3], drop_quat[0], drop_quat[1], drop_quat[2]
        )
    
    def _create_pose(self, x, y, z, qw, qx, qy, qz, frame_id="panda_link0"):
        """Create PoseStamped message"""
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        
        pose.pose.orientation.w = qw
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        
        return pose
    
    def _compute_approach_vector(self, pose):
        """
        Return the unit vector that points *into* the gripper
        (its local –Z axis) expressed in the world frame.
        """
        quat = np.array([pose.orientation.x,
                         pose.orientation.y,
                         pose.orientation.z,
                         pose.orientation.w])
        rot  = R_simple.from_quat(quat)  # SciPy/ROS use (x,y,z,w)
        local_z_neg = np.array([0.0, 0.0, 1.0])
        world_vec = rot.apply(local_z_neg)
        return world_vec / np.linalg.norm(world_vec)

    def _create_pre_grasp_pose(self, grasp_ps, distance):
        """
        Build a PoseStamped offset *opposite* the approach direction.
        Works for top-down and side grasps alike.
        """
        v = self._compute_approach_vector(grasp_ps.pose)

        pre = copy.deepcopy(grasp_ps)
        pre.header.stamp = self.get_clock().now().to_msg()
        pre.pose.position.x -= distance * v[0]
        pre.pose.position.y -= distance * v[1]
        pre.pose.position.z -= distance * v[2]
        return pre
    
    def _create_offset_pose(self, base_pose, z_offset=0.0):
        """Create pose with Z offset from base pose"""
        new_pose = PoseStamped()
        new_pose.header = base_pose.header
        new_pose.header.stamp = self.get_clock().now().to_msg()
        
        # Copy position with offset
        new_pose.pose.position.x = base_pose.pose.position.x
        new_pose.pose.position.y = base_pose.pose.position.y
        new_pose.pose.position.z = base_pose.pose.position.z + z_offset
        
        # Copy and validate orientation
        new_pose.pose.orientation = base_pose.pose.orientation
        
        return new_pose

    def _is_pose_reachable(self, pose):
        """Check if pose is within workspace and safe"""
        x, y, z = pose.position.x, pose.position.y, pose.position.z
        limits = self.config['workspace_limits']

        # Check workspace limits
        if not (limits['x_min'] <= x <= limits['x_max'] and
                limits['y_min'] <= y <= limits['y_max'] and
                limits['z_min'] <= z <= limits['z_max']):
            self.get_logger().warn(f"Pose [{x:.3f}, {y:.3f}, {z:.3f}] outside workspace")
            return False
    
        # # Check table collision
        # if self._check_table_collision(pose):
        #     self.get_logger().warn(f"Pose [{x:.3f}, {y:.3f}, {z:.3f}] would cause table collision")
        #     return False
        
        return True


    ##### IMPLENTATION IS WRONG, NEEDS TO BE FIXED #####
    def _check_table_collision(self, pose, table_height=0.0): 
        """Check if gripper will collide with table"""
        actual_pose = pose.pose if hasattr(pose, 'pose') else pose
        pos = actual_pose.position
        quat = np.array([
            actual_pose.orientation.x,
            actual_pose.orientation.y, 
            actual_pose.orientation.z,
            actual_pose.orientation.w
        ])
        
        # Calculate gripper angle
        euler = R_simple.from_quat(quat).as_euler('xyz', degrees=True)
        alpha_deg = euler[0] - 180.0
        alpha_rad = np.radians(alpha_deg)
        
        if abs(np.sin(alpha_rad)) < 0.01:
            return False  # Default position, assume safe
        
        # Physical parameters
        FINGER_HEIGHT = 0.018
        FINGER_WIDTH = 0.005
        SAFETY_MARGIN = 0.005
        half_gripper_width = self.config['gripper']['max_width'] / 2.0
        
        sin_alpha = np.sin(alpha_rad)
        finger_offset = FINGER_HEIGHT / (2 * abs(sin_alpha))
        z_min_safe = (abs(sin_alpha) * (half_gripper_width + FINGER_WIDTH + finger_offset) + 
                     SAFETY_MARGIN + table_height)
        
        clearance = pos.z - z_min_safe
        
        if clearance < 0:
            self.get_logger().warn(f"Table collision risk: clearance={clearance*1000:.1f}mm")
            return True
        
        return False
    #####################################################

    def _check_grasp_success(self):
        """Check if grasp was successful"""
        if self.gripper_width is None:
            raise GraspFailureException("Cannot verify grasp: no gripper data")
        
        grasp_failure_threshold = 0.005  # 5mm
        
        self.get_logger().info(f"Gripper width: {self.gripper_width*1000:.1f}mm")
        
        if self.gripper_width <= grasp_failure_threshold:
            raise GraspFailureException("Grasp failed: gripper closed without object")
        
        return True

    def _move_to_pose(self, pose, description="", timeout=8.0):
        """Send pose to cartesian impedance controller"""
        self.target_pose = pose
        self.cartesian_target_publisher.publish(pose)
        self.riemann_target_publisher.publish(pose)
        
        if description:
            self.get_logger().info(f"{description}: [{pose.pose.position.x:.3f}, {pose.pose.position.y:.3f}, {pose.pose.position.z:.3f}]")
        
        return True
    
    def _open_gripper(self):
        """Open gripper and return future"""
        self.get_logger().info(f"Opening gripper to {self.config['gripper']['max_width']*1000:.0f}mm")
        goal = Move.Goal()
        goal.width = self.config['gripper']['max_width']
        goal.speed = self.config['gripper']['speed']
        return self.move_client.send_goal_async(goal)
    
    def _close_gripper(self):
        """Close gripper and return future"""
        self.get_logger().info(f"Grasping with {self.config['gripper']['force']:.0f}N force")
        goal = Grasp.Goal()
        goal.width = self.config['gripper']['goal_width']
        goal.speed = self.config['gripper']['speed']
        goal.force = self.config['gripper']['force']
        goal.epsilon.inner = self.config['gripper']['epsilon_inner']
        goal.epsilon.outer = self.config['gripper']['epsilon_outer']
        return self.grasp_client.send_goal_async(goal)

    def _publish_status(self, success):
        """Publish grasp status"""
        msg = Bool()
        msg.data = success
        self.status_publisher.publish(msg)

    def _publish_execution_state(self, executing):
        """Publish execution state"""
        msg = Bool()
        msg.data = executing
        self.execution_state_publisher.publish(msg)

    def _on_grasp_pose_received(self, msg):
        """Handle new grasp pose"""
        self.get_logger().info(
            f"Received grasp pose: [{msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f}]"
        )
        
        if self.current_state != GraspState.IDLE:
            # Store as pending instead of overwriting current
            self.pending_grasp_pose = msg
            self.get_logger().info("Robot busy - pose stored as pending")
            return
        
        # If idle, start immediately
        if self._is_pose_reachable(msg.pose):
            self.active_grasp_pose = msg
            self._transition_to_state(GraspState.MOVING_TO_HOME)
        else:
            self.get_logger().error("Pose not reachable - ignoring")

    def _transition_to_state(self, new_state):
        """Transition to a new state"""
        old_state = self.current_state
        self.current_state = new_state
        self.state_start_time = time.time()
        self.current_future = None
        
        if new_state != GraspState.IDLE:
            self._publish_execution_state(True)
        else:
            self._publish_execution_state(False)
        
        self.get_logger().info(f"State: {old_state.value} -> {new_state.value}")

    def _check_timeout(self, timeout_seconds):
        """Check if current state has timed out"""
        if self.state_start_time is None:
            return False
        return (time.time() - self.state_start_time) > timeout_seconds

    def _state_machine_callback(self):
        """State machine timer callback"""
        try:
            if self.current_state == GraspState.IDLE:
                self._handle_idle_state()
            elif self.current_state == GraspState.MOVING_TO_HOME:
                self._handle_moving_to_home_state()
            elif self.current_state == GraspState.OPENING_GRIPPER:
                self._handle_opening_gripper_state()
            elif self.current_state == GraspState.MOVING_TO_PRE_GRASP:
                self._handle_moving_to_pre_grasp_state()
            elif self.current_state == GraspState.MOVING_TO_GRASP:
                self._handle_moving_to_grasp_state()
            elif self.current_state == GraspState.CLOSING_GRIPPER:
                self._handle_closing_gripper_state()
            elif self.current_state == GraspState.CHECKING_GRASP:
                self._handle_checking_grasp_state()
            elif self.current_state == GraspState.LIFTING_OBJECT:
                self._handle_lifting_object_state()
            elif self.current_state == GraspState.RETURNING_TO_HOME:
                self._handle_returning_to_home_state()
            elif self.current_state == GraspState.MOVING_TO_DROP_SAFE:
                self._handle_moving_to_drop_safe_state()
            elif self.current_state == GraspState.MOVING_TO_DROP:
                self._handle_moving_to_drop_state()
            elif self.current_state == GraspState.RELEASING_OBJECT:
                self._handle_releasing_object_state()
            elif self.current_state == GraspState.RETREATING_FROM_DROP:
                self._handle_retreating_from_drop_state()
            elif self.current_state == GraspState.FINAL_HOME:
                self._handle_final_home_state()
            elif self.current_state == GraspState.ERROR_RECOVERY:
                self._handle_error_recovery_state()
                
        except Exception as e:
            self.get_logger().error(f"State machine error: {e}")
            self._transition_to_state(GraspState.ERROR_RECOVERY)

    def _handle_idle_state(self):
        """Handle idle state"""
        # Check for pending poses
        if self.pending_grasp_pose and self._is_pose_reachable(self.pending_grasp_pose.pose):
            self.active_grasp_pose = self.pending_grasp_pose
            self.pending_grasp_pose = None
            self._transition_to_state(GraspState.MOVING_TO_HOME)

    def _handle_moving_to_home_state(self):
        """Home movement"""
        if self.current_future is None:
            self._move_to_pose(self.home_pose, "Moving to home position")
            self.current_future = True
        
        # Completion check
        status = self._movement_completed(8.0)
        if status ==  'completed':
            self._transition_to_state(GraspState.OPENING_GRIPPER)
        elif status == 'failed':
            self._transition_to_state(GraspState.ERROR_RECOVERY)

    def _handle_opening_gripper_state(self):
        """Handle opening gripper state"""
        if self.current_future is None:
            self.current_future = self._open_gripper()
        
        if self.current_future.done() or self._check_timeout(self.config['timeouts']['gripper']):
            self._transition_to_state(GraspState.MOVING_TO_PRE_GRASP)

    def _handle_moving_to_pre_grasp_state(self):
        """Handle moving to pre-grasp state with real feedback"""
        if self.current_future is None:
            pre_grasp_pose = self._create_pre_grasp_pose(
                self.active_grasp_pose, 
                self.config['offsets']['pre_grasp']
            )
            self._move_to_pose(pre_grasp_pose, "Moving to pre-grasp position", timeout=8.0)
            self.current_future = True
        
        status = self._movement_completed(8.0)
        if status == 'completed':
            self._transition_to_state(GraspState.MOVING_TO_GRASP)
        elif status == 'failed':
            self.get_logger().error("Failed to reach pre-grasp position")
            self._transition_to_state(GraspState.ERROR_RECOVERY)

    def _handle_moving_to_grasp_state(self):
        """Handle moving to grasp state"""
        if self.current_future is None:
            self.current_future = self._move_to_pose(self.active_grasp_pose, "Moving to grasp position", timeout=8.0)

        status = self._movement_completed(8.0)
        if status == 'completed':
            self._transition_to_state(GraspState.CLOSING_GRIPPER)
        elif status == 'failed':
            self.get_logger().error("Failed to reach grasp position")
            self._transition_to_state(GraspState.ERROR_RECOVERY)

    def _handle_closing_gripper_state(self):
        """Handle closing gripper state"""
        if self.current_future is None:
            self.current_future = self._close_gripper()
        
        if self.current_future.done() or self._check_timeout(self.config['timeouts']['gripper']):
            self._transition_to_state(GraspState.CHECKING_GRASP)

    def _handle_checking_grasp_state(self):
        """Handle checking grasp state"""
        try:
            if self._check_timeout(3.0):  # 3 seconds for grasp check
                self.grasp_successful = self._check_grasp_success()
                self._transition_to_state(GraspState.LIFTING_OBJECT)
        except GraspFailureException as e:
            self.get_logger().error(f"Grasp failed: {e}")
            self.grasp_successful = False
            self._transition_to_state(GraspState.ERROR_RECOVERY)

    def _handle_lifting_object_state(self):
        """Handle lifting object state"""
        if self.current_future is None:
            lift_pose = self._create_offset_pose(self.active_grasp_pose, self.config['offsets']['lift'])
            self.current_future = self._move_to_pose(lift_pose, "Lifting object", timeout=8.0)

        status = self._movement_completed(8.0)
        if status == 'completed':
            self._transition_to_state(GraspState.RETURNING_TO_HOME)
        elif status == 'failed':
            self.get_logger().error("Failed to lift object")
            self._transition_to_state(GraspState.ERROR_RECOVERY)

    def _handle_returning_to_home_state(self):
        """Handle returning to home state"""
        if self.current_future is None:
            self.current_future = self._move_to_pose(self.home_pose, "Returning to home with object", timeout=8.0)

        status = self._movement_completed(8.0)
        if status == 'completed':
            self._transition_to_state(GraspState.MOVING_TO_DROP_SAFE)
        elif status == 'failed':
            self.get_logger().error("Failed to return to home position")
            self._transition_to_state(GraspState.ERROR_RECOVERY)

    def _handle_moving_to_drop_safe_state(self):
        """Handle moving to drop safe state"""
        if self.current_future is None:
            drop_safe_pose = self._create_offset_pose(self.drop_pose, self.config['offsets']['drop_safe'])
            self.current_future = self._move_to_pose(drop_safe_pose, "Moving to drop safe position", timeout=8.0)

        status = self._movement_completed(8.0)
        if status == 'completed':
            self._transition_to_state(GraspState.MOVING_TO_DROP)
        elif status == 'failed':
            self.get_logger().error("Failed to reach drop safe position")
            self._transition_to_state(GraspState.ERROR_RECOVERY)

    def _handle_moving_to_drop_state(self):
        """Handle moving to drop state"""
        if self.current_future is None:
            self.current_future = self._move_to_pose(self.drop_pose, "Moving to drop position", timeout=8.0)

        status = self._movement_completed(8.0)
        if status == 'completed':
            self._transition_to_state(GraspState.RELEASING_OBJECT)
        elif status == 'failed':
            self.get_logger().error("Failed to reach drop position")
            self._transition_to_state(GraspState.ERROR_RECOVERY)

    def _handle_releasing_object_state(self):
        """Handle releasing object state"""
        if self.current_future is None:
            self.current_future = self._open_gripper()
        
        if self.current_future.done() or self._check_timeout(self.config['timeouts']['gripper']):
            self._transition_to_state(GraspState.RETREATING_FROM_DROP)

    def _handle_retreating_from_drop_state(self):
        """Handle retreating from drop state"""
        if self.current_future is None:
            drop_approach_pose = self._create_offset_pose(self.drop_pose, self.config['offsets']['drop_approach'])
            self.current_future = self._move_to_pose(drop_approach_pose, "Retreating from drop position", timeout=8.0)

        status = self._movement_completed(8.0)
        if status == 'completed':
            self._transition_to_state(GraspState.FINAL_HOME)
        elif status == 'failed':
            self.get_logger().error("Failed to retreat from drop position")
            self._transition_to_state(GraspState.ERROR_RECOVERY)

    def _handle_final_home_state(self):
        """Handle final home state"""
        if self.current_future is None:
            self.current_future = self._move_to_pose(self.home_pose, "Returning to final home position", timeout=8.0)

        status = self._movement_completed(8.0)
        if status == 'completed':
            # Sequence completed successfully
            self._publish_status(self.grasp_successful)
            
            if self.grasp_successful:
                self.get_logger().info("Sequence completed successfully!")
            else:
                self.get_logger().info("Sequence completed but grasp may have failed")
            
            # Clear all poses and return to idle
            self.active_grasp_pose = None
            self.pending_grasp_pose = None
            self.grasp_successful = False
            self.get_logger().info("Cleared all poses - waiting for NEW object detection")
            
            self._transition_to_state(GraspState.IDLE)

    def _handle_error_recovery_state(self):
        """Handle error recovery state"""
        if self.current_future is None:
            self.get_logger().info("Starting error recovery...")
            self.current_future = self._open_gripper()
        
        # First ensure gripper is open
        if self.current_future.done() or self._check_timeout(self.config['timeouts']['gripper']):
            # Then move to home
            if not hasattr(self, '_recovery_home_started'):
                self._recovery_home_started = True
                self.current_future = self._move_to_pose(self.home_pose, "Error recovery - returning home")
            elif self._check_timeout(8.0):  # Total recovery timeout
                # Recovery complete
                self._publish_status(False)
                self.active_grasp_pose = None
                self.pending_grasp_pose = None
                self.grasp_successful = False
                delattr(self, '_recovery_home_started')
                self.get_logger().info("Error recovery completed")
                self._transition_to_state(GraspState.IDLE)
                
    def _log_pose_status(self):
        """Log current pose status for debugging"""
        if self.current_robot_pose and self.target_pose:
            pos_diff = np.sqrt(
                (self.current_robot_pose.pose.position.x - self.target_pose.pose.position.x)**2 +
                (self.current_robot_pose.pose.position.y - self.target_pose.pose.position.y)**2 +
                (self.current_robot_pose.pose.position.z - self.target_pose.pose.position.z)**2
            )
            
            self.get_logger().debug(f"Position error: {pos_diff*1000:.1f}mm")
            
            if pos_diff > 0.05:  # Log if more than 5cm away
                current_pos = self.current_robot_pose.pose.position
                target_pos = self.target_pose.pose.position
                self.get_logger().info(
                    f"Current: [{current_pos.x:.3f}, {current_pos.y:.3f}, {current_pos.z:.3f}] "
                    f"Target: [{target_pos.x:.3f}, {target_pos.y:.3f}, {target_pos.z:.3f}]"
                )
                
    def _check_robot_state_valid(self):
        """Check if robot state is valid and safe"""
        if self.current_robot_pose is None:
            return False, "No robot pose feedback"
        
        # Check if pose is recent (within last 500ms)
        current_time = self.get_clock().now()
        pose_time = rclpy.time.Time.from_msg(self.current_robot_pose.header.stamp)
        age = (current_time - pose_time).nanoseconds / 1e9
        
        if age > 0.5:  # 500ms threshold
            return False, f"Robot pose too old ({age:.1f}s)"
        
        # Check if robot is in workspace
        pos = self.current_robot_pose.pose.position
        if not self._is_pose_reachable(self.current_robot_pose.pose):
            return False, f"Robot outside workspace: [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]"
        
        return True, "Robot state valid"


def main(args=None):
    rclpy.init(args=args)
    node = GraspExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()