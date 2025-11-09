import time
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
from pathlib import Path

# ============================
#   CONFIGURATION
# ============================
simulation = True
APPROACH_HEIGHT = 0.1

# --- Load robot ---
def find_urdf(model_dir: Path):
    if not model_dir.exists():
        raise FileNotFoundError(f"Folder tidak ditemukan: {model_dir}")
    candidates = list(model_dir.rglob("*.urdf"))
    if not candidates:
        raise FileNotFoundError(f"Tidak ada .urdf di: {model_dir}")
    for c in candidates:
        if c.name.lower() == "robot.urdf":
            return c
    return candidates[0]

MODEL_DIR = Path.home() / "Documents" / "Code" / "inverse-kinematics" / "hiro-urdf"
urdf_path = find_urdf(MODEL_DIR)

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    str(urdf_path), str(MODEL_DIR)
)
data = model.createData()

viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()
viz.displayVisuals(True)

# --- Initial pose: kaki siap dari atas ---
q0 = pin.neutral(model)

hip = model.getJointId("left_hip_pitch")
knee = model.getJointId("left_knee")
ankle = model.getJointId("left_ankle_pitch")

# if hip:
#     q0[model.joints[hip].idx_q] += 0.4
# if knee:
#     q0[model.joints[knee].idx_q] += 0.7
# if ankle:
#     q0[model.joints[ankle].idx_q] -= 0.5

q = q0.copy()
viz.display(q)

ball_pos = np.array([0.3, 0.0, -0.2])
viz.viewer["/ball"].set_object(
    g.Sphere(0.05),
    g.MeshLambertMaterial(color=0xff0000, opacity=0.7)
)
T_ball = np.eye(4)
T_ball[:3, 3] = ball_pos
viz.viewer["/ball"].set_transform(T_ball)

if not simulation:
    exit()
    
joint_limits = {
    'left_ankle_pitch': (np.deg2rad(-75), np.deg2rad(75)),
    'left_hip_pitch': (np.deg2rad(-60), np.deg2rad(60)),
    'left_knee': (np.deg2rad(-90), np.deg2rad(90))
}

def apply_joint_limits(q):
    q_clamped = q.copy()
    for joint_name, (q_min, q_max) in joint_limits.items():
        jid_val = model.getJointId(joint_name)
        if jid_val is not None:
            idx = model.joints[jid_val].idx_q
            q_clamped[idx] = np.clip(q[idx], q_min, q_max)
    return q_clamped

ee_frame = model.getFrameId('left_foot_frame')
dt = 0.04
damping = 1e-3
eps = 1e-4
max_iter = 15

def generateApproach():
    trajectory = []
    
    n_approach = 50
    for i in range(n_approach):
        t = i / n_approach
        s = t * t * (3 - 2*t)
        
        pos = ball_pos + np.array([
            -0.15 * (1-s),  
            0,
            APPROACH_HEIGHT * (1-s)  
        ])
        
        rot = pin.utils.rotate('y', -np.pi/6 * (1-s)) @ pin.utils.rotate('z', np.pi/2)
        trajectory.append((pos, rot))
    
    for i in range(15):
        pos = ball_pos.copy()
        rot = pin.utils.rotate('z', np.pi/2)
        trajectory.append((pos, rot))
    
    return trajectory

trajectory = generateApproach()
print("Running approach motion... (Ctrl+C to stop)")

try:
    idx = 0
    while True:
        target_pos, target_rot = trajectory[idx % len(trajectory)]
        target_SE3 = pin.SE3(target_rot, target_pos)
        
        offset = pin.SE3(np.eye(3), np.array([0, -0.1, -0.03]))
        
        for _ in range(max_iter):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            oMf = data.oMf[ee_frame] * offset
            
            err = pin.log(target_SE3.inverse() * oMf).vector
            if np.linalg.norm(err) < eps:
                break
            
            J = pin.computeFrameJacobian(model, data, q, ee_frame, pin.ReferenceFrame.LOCAL)
            dq = -np.linalg.solve(J.T @ J + damping * np.eye(model.nv), J.T @ err)
            q = pin.integrate(model, q, dq)
            
            q = apply_joint_limits(q)
        
        viz.display(q)
        
        idx += 1
        if idx >= len(trajectory):
            time.sleep(1.0)  
            idx = 0
            q = q0.copy()
        
        time.sleep(dt)

except KeyboardInterrupt:
    print("\nStopped.")
