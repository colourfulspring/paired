### create_parallel_env函数（以minigrid/paired.json为参数）

问题：
* 生成环境实例env和ued_venv。二者有何区别？
    - 二者仅仅在用于生成minigrid图像的VecPreprocessImageWrapper上参数不同，以及venv被重新设置了随机数种子而ued_venv没有。是同一个ParallelAdversarialEnv实例（可能套Wrapper，下文均称ParallelAdversarialEnv）的不同名字。ParallelAdversarialEnv维护了num_envs个子进程，每个进程包含一个GoalLastAdversarialEnv实例，该Env继承了multigrid.MultigridEnv提供的Protagonist的step API，同时也继承了AdversarialEnv提供的Adversary的step_adversary API。这说明venv和ued_venv二者既支持Protagonist调用step在环境中交互，也支持Adversary调用step_adversary对环境元素进行放置。**一定要使用两个名字的原因目前还在调查之中，应该不会仅仅是上面讲的这些**。

输入输出：
* 输入：ArgumentParser生成的参数args。
* 输出：两个ParallelAdversarialEnv套了一些VecEnvWrapper的子类实例：venv，ued_venv。

继承链：
* gym.Env&rarr;minigrid.MiniGridEnv&rarr;multigrid.MultiGridEnv&rarr;AdversarialEnv&rarr;GoalLastAdversarialEnv (Minigrid Adversary环境底层实现)
* VecEnv&rarr;SubprocVecEnv&rarr;ParallelAdversarialEnv (并行环境逻辑类，环境是原型)
* VecEnv&rarr;VecEnvWrapper&rarr;VecPreprocessImageWrapper, VecMonitor, VecNormalize（并行环境的Wrapper系列）

函数执行步骤：
> :bulb: **Input:** ArgumentParser生成的参数args。

1. 构造元素为lambda表达式`_make_env(args)`，长度32的`list`。

2. 调用一个ParallelAdversarialEnv实例，包含len(env_fns)个子进程。每个子进程包含一个GoalLastAdversarialEnv实例。进程之间通过管道通信。
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

3. 根据venv内部的并行envs是否为minigrid，选择合适的参数，并给ued_venv套上VecPreprocessImageWrapper。
4. 给venv设随机数种子。
> :bulb: **Output:** 两个ParallelAdversarialEnv套了一些VecEnvWrapper的子类实例：venv，ued_venv。

### make_agent函数

函数执行步骤：
> :bulb: **Input:** 智能体名字name=, ParallelAdversarialVecEnv的实例venv，ArgumentParser生成的实参args。
1. 从name得知该智能体不是对手环境，从而获取venv的action_space和observation_space以及其他构建网络需要的参数。
2. 调用model_for_env_agent函数，得到网络actor_critic。
3. 根据网络actor_critic以及实参args的一部分生成PPO类实例algo，RolloutStorage类storage。
4. 将algo和storage类组合为ACAgent类agent。

### MultigridNetwork类（name='agent'）

问题：
1. 该网络的结构如何？输入输出的维度大小是什么？表示什么意义？

该网络的包含下面这些submodule：
```
OrderedDict([('image_conv', Sequential(
  (0): Conv2d_tf(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=VALID)
  (1): Flatten(start_dim=1, end_dim=-1)
  (2): ReLU()
)), ('scalar_embed', Linear(in_features=4, out_features=5, bias=True)), ('rnn', RNN(
  (rnn): LSTM(149, 256)
)), ('actor', Sequential(
  (0): Sequential(
    (0): Linear(in_features=256, out_features=32, bias=True)
    (1): Tanh()
  )
  (1): Categorical(
    (linear): Linear(in_features=32, out_features=7, bias=True)
  )
)), ('critic', Sequential(
  (0): Sequential(
    (0): Linear(in_features=256, out_features=32, bias=True)
    (1): Tanh()
  )
  (1): Linear(in_features=32, out_features=1, bias=True)
))])
```
Categorical：包含了一个nn.Linear和一个distribution.Categorical，根据线性层的分布采样输出。

输入(5,5,3)的image，输出(1,)的动作和(1,)价值。

模型如何拼接的没看出来。

### MultigridNetwork类（name='adversary_env'）
问题：
1. 该网络的结构如何？输入输出的维度大小是什么？表示什么意义？
该网络包含下面这些submodule:
```
OrderedDict([('image_conv', Sequential(
  (0): Conv2d_tf(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=VALID)
  (1): Flatten(start_dim=1, end_dim=-1)
  (2): ReLU()
)), ('scalar_embed', Linear(in_features=53, out_features=10, bias=True)), ('rnn', RNN(
  (rnn): LSTM(21692, 256)
)), ('actor', Sequential(
  (0): Sequential(
    (0): Linear(in_features=256, out_features=32, bias=True)
    (1): Tanh()
  )
  (1): Categorical(
    (linear): Linear(in_features=32, out_features=169, bias=True)
  )
)), ('critic', Sequential(
  (0): Sequential(
    (0): Linear(in_features=256, out_features=32, bias=True)
    (1): Tanh()
  )
  (1): Linear(in_features=32, out_features=1, bias=True)
))])
```
输入输出待定。

模型如何拼接的没看出来。

2. 两个网络有什么区别？

### RolloutStorage类
大概是用于存观测到的数据的。具体如何存没看出来。

### AdversarialRunner类
> :bulb: **Input:** 实参args。两个并行的训练环境venv, ued_venv，支持Protagonist和Adversary双方API。三个待训练的智能体Protagonist，Antagonist和Adversary enviroment。

### Evaluator类

### 其他
* zip函数将元组的列表转换为列表的元组。可以通过赋值和迭代两种方法取出元素。
* 创建一个新的Process时会深拷贝传入给worker的参数。