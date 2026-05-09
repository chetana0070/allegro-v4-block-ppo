import mujoco
from envs.allegro_block_env import AllegroBlockEnv

env = AllegroBlockEnv()
obs, info = env.reset()

model = env.model
data = env.data

print("ncon:", data.ncon)

for i in range(data.ncon):
    c = data.contact[i]

    g1 = c.geom1
    g2 = c.geom2

    g1_name = model.geom(g1).name
    g2_name = model.geom(g2).name

    b1 = model.body(model.geom_bodyid[g1]).name
    b2 = model.body(model.geom_bodyid[g2]).name

    print(
        f"{i:02d}: "
        f"geom=({g1_name}, {g2_name}) "
        f"body=({b1}, {b2}) "
        f"dist={c.dist:.6f}"
    )
