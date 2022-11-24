from gym.envs.registration import register

register(
    id='object_manipulation_task_task1-v0',
    entry_point='gym_object_manipulation_task.envs:object_manipulation_task_task1Env',
    max_episode_steps=300,
)

register(
    id='object_manipulation_task_task2-v0',
    entry_point='gym_object_manipulation_task.envs:object_manipulation_task_task2Env',
    max_episode_steps=300,
)

register(
    id='object_manipulation_task_composite_task-v0',
    entry_point='gym_object_manipulation_task.envs:object_manipulation_task_composite_taskEnv',
    max_episode_steps=300,
)
