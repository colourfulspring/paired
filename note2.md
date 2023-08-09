### create_parallel_env函数（以minigrid/paired.json为参数）

问题：
* 生成环境实例env和ued_venv。二者有何区别？
    - 没有区别。二者仅仅是同一个ParallelAdversarialEnv实例（可能套Wrapper，下文均称ParallelAdversarialEnv）的不同名字。ParallelAdversarialEnv维护了num_envs个子进程，每个进程包含一个GoalLastAdversarialEnv实例，该Env继承了multigrid.MultigridEnv提供的Protagonist的step API，同时也继承了AdversarialEnv提供的Adversary的step_adversary API。使用两个名字的原因还在调查之中。

输入输出：
* 输入：ArgumentParser生成的参数args。
* 输出：两个ParallelAdversarialEnv套了一些VecEnvWrapper的子类实例：venv，ued_venv。

继承链：
* gym.Env&rarr;minigrid.MiniGridEnv&rarr;multigrid.MultiGridEnv&rarr;AdversarialEnv&rarr;GoalLastAdversarialEnv (Minigrid Adversary环境底层实现)
* VecEnv&rarr;SubprocVecEnv&rarr;ParallelAdversarialEnv (并行环境逻辑类，环境是原型)
* VecEnv&rarr;VecEnvWrapper&rarr;VecPreprocessImageWrapper, VecMonitor, VecNormalize（并行环境的Wrapper系列）

函数执行步骤：
> :bulb: **Input:** lambda表达式`_make_env(args)`的`list`。

1. 调用一个ParallelAdversarialEnv实例，包含len(env_fns)个子进程。每个子进程包含一个GoalLastAdversarialEnv实例。进程之间通过管道通信。
> :bulb: **Output:** ParallelAdversarialVecEnv实例`venv`。
* venv：
    - ParallelAdversarialVecEnv
        - **action_dims:** 1。Protagonist动作维度。
        - **adversary_action_dims:** 1。Adversary动作维度。
    - &rarr; SubprocVecEnv
        - **num_envs:** 32。 并行环境数。
        - **ps:** 子进程列表。
        - **remotes:** 与子进程的连接列表。

* envs: GoalLastAdversarialEnv (`in subprocess`)
    - **action_space:**  Discrete(7)。  环境接受的Protagonist的7个动作。
    - **observation_space:**  Dict(direction:Box(1,), image:Box(5, 5, 3))。 环境提供的Protagonist的观测。
    - **adversary_action_space:** Discrete(169)。环境接受的Antagonist的169个动作，选择摆放wall的位置。
    - **adversary_observation_space:** Dict(image:Box(15, 15, 3), random_z:Box(50,), time_step:Box(1,))。环境提供的Protagonist的观测。
    - **max_steps:** 250。Protagonist最多执行的动作步数。
    - **choose_goal_last:** True。与Adversariy约定摆放顺序为先Wall后Agent和Goal。
    - **n_agents:** 1。单智能体。
    - **n_clutter:** 50。最多摆放Wall数目。
    - **adversary_max_steps:** 52。Adversary最多执行的动作步数。最多摆放Wall数目+一个Agent+一个Goal。

> :warning: **Warning:** GoalLastAdversarialEnv提供的observation_space不一定都被使用了，真正被使用的observation应该看神经网络的输入。action_space则相反，是神经网络的输出不一定被使用了，真正被使用的action应该看step或step_adversary的输入。


2. 




### 其他
* zip函数将元组的列表转换为列表的元组。可以通过赋值和迭代两种方法取出元素。
* 创建一个新的Process时会深拷贝传入给worker的参数。