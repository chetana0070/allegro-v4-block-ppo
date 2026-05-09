from envs.allegro_block_env import AllegroBlockEnv

env = AllegroBlockEnv()
obs, info = env.reset()

print("XML + ENV LOAD OK")
print("obs shape:", obs.shape)
print("action shape:", env.action_space.shape)
print("action low:", env.action_space.low[:4])
print("action high:", env.action_space.high[:4])

for i in range(50):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)

    print(
        f"step={i:03d} "
        f"z={info['lift_height']:.4f} "
        f"finger_dist={info['finger_dist']:.4f} "
        f"move={info['block_move']:.5f} "
        f"reward={reward:.3f} "
        f"success={info['success']}"
    )

    if term or trunc:
        break

print("SMOKE TEST DONE")
