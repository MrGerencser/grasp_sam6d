import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R_simple

class Superquadric:
    def __init__(self, ε, a, euler, t):
        # ε = [ε1, ε2]; a = [ax, ay, az]; euler = [roll, pitch, yaw]; t = [tx, ty, tz]
        self.ε1, self.ε2 = ε
        self.ax, self.ay, self.az = a
        self.T = np.asarray(t, dtype=float)        # Center in world frame
        # 3×3 rotation matrix from SQ‐local → world
        self.R = R_simple.from_euler('xyz', euler).as_matrix()

    @property
    def axes_world(self):
        """
        Return the three principal axes (λ_x, λ_y, λ_z) as columns in world frame.
        """
        return self.R  # each column is a unit vector
    
class Gripper:
    """
    Parallel-jaw gripper description **in its own local frame**.

    • +Y  – closing line  (jaws move ±Y)   ← matches paper
    • –Z  – approach axis (tool moves –Z) ← matches paper
    •  X  – completes the RH frame
    """

    def __init__(self,
                 jaw_len   = 0.041,   # finger length  (m)
                 max_open  = 0.080,   # maximum jaw separation (m)
                 thickness = 0.004,   # finger thickness l_j (m)
                 palm_depth= 0.02,   # distance from jaw mid-line to tool flange (m)
                 palm_width= 0.04):  # width of the aluminium bracket (m)

        # --- geometry used by the paper’s tests --------------------
        self.jaw_len      = float(jaw_len)
        self.max_open     = float(max_open)
        self.thickness    = float(thickness)   # l_j  (support & collision radius)

        # --- local coordinate conventions --------------------------
        self.lambda_local = np.array([0., 1., 0.])   # closing line
        self.approach_axis= np.array([0., 0., -1.])  # approach

        # --- extras for viz or URDF generation ---------------------
        self.palm_depth   = float(palm_depth)
        self.palm_width   = float(palm_width)

    # ---------- helper: analytic Open3D meshes ---------------------
    def make_open3d_meshes(self, colour=(0.2, 0.8, 0.2)):
        """
        Returns four cylinders in gripper-local coordinates:
            finger_L, finger_R, cross_bar_Y, back_Z
        All share the same radius = thickness and specified colour.
        """
        meshes = []

        # 1. LEFT FINGER (+Y) - create with correct dimensions directly
        finger_L = o3d.geometry.TriangleMesh.create_cylinder(
            radius=self.thickness,      # Keep original thickness
            height=self.jaw_len         # Set correct height directly
        )
        finger_L.paint_uniform_color(colour)
        finger_L.compute_vertex_normals()
        
        # Position left finger
        T = np.eye(4)
        T[1, 3] = +self.max_open / 2 + self.thickness   # y-offset
        T[2, 3] = -self.jaw_len / 2     # so tip sits at Z = 0
        finger_L.transform(T)
        meshes.append(finger_L)

        # 2. RIGHT FINGER (-Y) - create with correct dimensions directly
        finger_R = o3d.geometry.TriangleMesh.create_cylinder(
            radius=self.thickness,      # Keep original thickness
            height=self.jaw_len         # Set correct height directly
        )
        finger_R.paint_uniform_color(colour)
        finger_R.compute_vertex_normals()
        
        # Position right finger
        T = np.eye(4)
        T[1, 3] = -self.max_open / 2 - self.thickness  # y-offset
        T[2, 3] = -self.jaw_len / 2
        finger_R.transform(T)
        meshes.append(finger_R)

        # 3. CROSS-BAR (axis = Y) - create with correct dimensions directly
        cross_Y = o3d.geometry.TriangleMesh.create_cylinder(
            radius=self.thickness,                      # Keep original thickness
            height=self.max_open + 4 * self.thickness       # Span across fingers
        )
        cross_Y.paint_uniform_color(colour)
        cross_Y.compute_vertex_normals()
        
        # Rotate so cylinder's axis (default +Z) becomes +Y
        R_x_neg90 = np.array([
            [1,  0,  0],
            [0,  0,  1],
            [0, -1,  0]
        ])
        T = np.eye(4)
        T[:3, :3] = R_x_neg90
        T[2, 3] = -self.jaw_len  # Place at finger tips (Z = -jaw_len)
        cross_Y.transform(T)
        meshes.append(cross_Y)

        # 4. BACK CYLINDER (axis = Z) - create with correct dimensions directly
        back_Z = o3d.geometry.TriangleMesh.create_cylinder(
            radius=self.thickness,      # Keep original thickness
            height=self.jaw_len      # Set correct height directly
        )
        back_Z.paint_uniform_color(colour)
        back_Z.compute_vertex_normals()
        
        # Position back cylinder
        T = np.eye(4)
        T[1, 3] = 0  # Y = 0 (centered between fingers)
        T[2, 3] = -3/2*self.jaw_len  # Behind fingers
        back_Z.transform(T)
        meshes.append(back_Z)

        return meshes

def rotation_from_u_to_v(u, v):
    """
    Compute the shortest‐arc rotation matrix that sends unit‐vector u → unit‐vector v.
    Handles the case u ≈ -v by picking an arbitrary perpendicular axis for a 180° spin.
    """
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    dot = np.dot(u, v)
    if dot > 1 - 1e-8:
        return np.eye(3)
    if dot < -1 + 1e-8:
        # u ≈ -v: pick arbitrary perpendicular axis
        if abs(u[0]) < 0.9:
            perp = np.array([1.0, 0.0, 0.0])
        else:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(u, perp)
        axis = axis / np.linalg.norm(axis)
        return R_simple.from_rotvec(axis * np.pi).as_matrix()
    axis = np.cross(u, v)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    return R_simple.from_rotvec(axis * angle).as_matrix()