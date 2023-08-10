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
> :bulb: **Input:** 智能体名字name, ParallelAdversarialVecEnv的实例venv，ArgumentParser生成的实参args。
1. 从name得知该智能体不是对手环境，从而获取venv的action_space和observation_space以及其他构建网络需要的参数。
2. 调用model_for_env_agent函数，得到网络actor_critic。
3. 根据网络actor_critic以及实参args的一部分生成PPO类实例algo，RolloutStorage类storage。
4. 将algo和storage类组合为ACAgent类agent。

### MultigridNetwork类（默认为name='agent'或'adversary_agent'，若有括号则括号中为name='adversary_env'）

问题：
1. 该网络的结构如何？输入输出的维度大小是什么？表示什么意义？


* actor_critic:
    - MultigridNetwork
        - **action_dim:** 1。似乎没用到。
        - **conv_kernel_size:** 3。模型卷积核size。
        - **conv_filters:** 16。模型卷积通道数。(128)
        - **image_conv:** 模型卷积头。将(3,5,5)输入张量卷积+拉直+ReLU。
        ```text
        Sequential(
            (0): Conv2d_tf(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=VALID)
            (1): Flatten(start_dim=1, end_dim=-1)
            (2): ReLU()
        )
        ```
        ```
        Sequential(
            (0): Conv2d_tf(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=VALID)
            (1): Flatten(start_dim=1, end_dim=-1)
            (2): ReLU()
        )
        ```
        
        - **image_embedding_size:** 144。16通道卷积核对(3,5,5)张量做卷积操作，得到(3\*3)\*9=144维。
        - **scalar_fc:** 5。标量嵌入后的维度。**具体干什么的还有待考证**。
        - **scalar_embed:** 模型标量嵌入头。将(4,)输入通过线性层转化为(5,)输出。
        ```
        Linear(in_features=4, out_features=5, bias=True)
        ```
        - **preprocessed_input_size:** 149。RNN输入层size。包含被image_conv卷积一次后的144维和嵌入后的标量5维。
        - **recurrent_hidden_size:** 256。RNN隐藏层size。
        - **recurrent_arch:** 'lstm'。RNN结构。
        - **rnn:** RNN网络。两个参数分别是网络的input_size和hidden_size。
        ```
        RNN(
            (rnn): LSTM(149, 256)
        )
        ```
        - **actor_fc_layers:** (32, 32)。动作头全连接层的维度。
        - **actor:** 动作头。Categorical是带离散动作采样的线性层。
        ```
        Sequential(
        (0): Sequential(
            (0): Linear(in_features=256, out_features=32, bias=True)
            (1): Tanh()
        )
        (1): Categorical(
            (linear): Linear(in_features=32, out_features=7, bias=True)
        )
        )
        ```
        - **critic_fc_layers:** (32, 32)。价值头全连接层的维度。
        - **critic:** 价值头。

        ```
        Sequential(
        (0): Sequential(
            (0): Linear(in_features=256, out_features=32, bias=True)
            (1): Tanh()
        )
        (1): Linear(in_features=32, out_features=1, bias=True)
        )
        ```

### MultigridNetwork类（name='adversary_env'）
问题：
1. 该网络的结构如何？输入输出的维度大小是什么？表示什么意义？


2. 两个网络有什么区别？

### PPO类
algo： 
参数为模型actor_critic和PPO算法需要的其他超参数。

### RolloutStorage类
用于存储多个时间步多个进程，智能体在环境中交互的数据。
* storage：(num_steps = 256, num_processes = 32)
    - **obs:** 一个和observation_space同构的字典，存储智能体观测。也有'direction'和'image'2个键。所有值在observation_space原维度基础上扩充2个新维度(num_steps + 1, num_processes, ...)，表示时间步和进程。
    - **recurrent_hidden_state_buffer_size:** 隐藏层buffer大小，与RNN结构有关。
    - **recurrent_hidden_states:** 一个(num_steps + 1, num_processes, recurrent_hidden_state_buffer_size)的张量。存储RNN的隐藏层状态。
    - **rewards, value_preds, returns:** 存储其他量的(num_steps + 1, num_processes, 1)张量。
    - **action_log_dist, action_log_probs, actions:** 存储动作相关量的(num_steps + 1, num_processes, ...)张量。
    - **level_seeds:** 存储随机数种子的(num_steps + 1, num_processes, 1)张量。

### ACAgent类
* agent: 包含algo和storage。
* adversary_agent：包含algo和storage。以上两者相同

### AdversarialRunner类
> :bulb: **Input:** 实参args。两个并行的训练环境venv, ued_venv，支持Protagonist和Adversary双方API。三个待训练的智能体Protagonist，Antagonist和Adversary enviroment。

### Evaluator类

### 其他
* zip函数将元组的列表转换为列表的元组。可以通过赋值和迭代两种方法取出元素。
* 创建一个新的Process时会深拷贝传入给worker的参数。