import mujoco
from mujoco import viewer
import numpy as np
import time

def main():
    model = mujoco.MjModel.from_xml_path("brax/envs/assets/classic_IP.xml")
    data = mujoco.MjData(model)

    data.qpos[0] = 0.1   # e.g., ~5.7 degrees tilt
    data.qvel[0] = 0.0   # no initial angular velocity

    viewer.launch(model, data)

    for _ in range(2000):
        mujoco.mj_step(model, data)
        if not viewer.is_running():
            break
        viewer.sync()
        time.sleep(0.01)  

if __name__ == "__main__":
    main()
