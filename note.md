# 目标和工作日志
1. 把minigrid中的adversarial env改成多智能体版本并实例化最简单的GoalLastAdversarialEnv的多智能体版本，将接入的PPO算法视为Centralized Training Centralized Execution结构训练多智能体中心控制器，能完成训练并渲染一个多智能体的迷宫的图：
    * [commit][doing] AdversarialEnv的n_agent=1，将其改为n_agent是参数的版本。新建类TwoAgentGoalLastAdversarialEnv(AdeversarialEnv)并用register函数注册为MultiGrid-TwoAgentGoalLastAdversarial-v0，含2个智能体。该类的observation_space和action_space调用gym库里的组合函数完成。AdversarialEnv的agent_start_pos只有单智能体版本，reset里只放了一个智能体，step_adversary的写法也只考虑了单智能体。
    * ~~[commit][planning][cancelled for Reason1] 修改MultiGridNetwork，使其支持动作空间dtype为int的离散Box类型。MultiGrid的动作不可能连续，所以可以不管连续情况。~~
    * [commit][planning]修改MultiGridNetwork，使其支持上一步生成的多智能体observation_space和action_space。
    * [test][planning] 测试是否可以训练10000步并生成多智能体的迷宫图。

* 问题：
    * `'Box' has no attribute 'n'`[solved]： ~~[cancelled for Reason1]MiniGrid多智能体情况动作空间是Box还是MultiDiscrete? (应选择Box。Discrete表示单智能体；Box将同样动作做笛卡尔积，相当于扩展；MultiDiscrete是多个不同的Discrete动作的拼接，比如遥控器的多个按钮，但也可以强行用来表示多个智能体。)在multigrid_models.py 37行增加对整型Box的支持。附近的两个self.num_action_logits似乎没有被使用，注释掉。~~
    * `AttributeError: 'RolloutStorage' object has no attribute 'action_log_dist'`[solved]：~~[cancelled for Reason1]RolloutStorage.action_log_dist就是MultigridNetwork.num_action_logits，已有的都能对上。在storage.py 82行增加对整型Box的支持。~~ 
    * `assert self.agent_pos[a] is not None`：adversarial.py的189行，在放置的时候只在start_pos处放了一个智能体。而应该生成多个位置，调用多个智能体。reset_agent，step_adversary方法。
* 原因：
    1. 这两个问题都可以归结到多智能体环境的observation_space和action_space在Network一端没有支持。重点是怎么解决多智能体的gym.Space。原项目重写了gym的EnvRegistry和EnvSpec，因为作者不希望gym自带的Atari，Mujoco等环境在import gym的时候被注册。还重写了gym的VecEnv，SyncVecEnv和AsyncVecEnv等并行环境容器，原因是gym自带的AsyncVecEnv库支持的命令太少了。其他功能都可以沿用gym库的代码，将单智能体Space构成多智能体Space可以沿用gym库的代码，不用自己研究。另外，也可以选择MultigridEnv类中的处理方法，对单智能体-多智能体分类解决。怎么解决还没想好。
2. 在1的前提下，修改ACAgent类，替换PPO算法为MAPPO算法，升级RolloutStorage功能，根据MAPPO修改MultigridNetwork网络结构。能完成训练并渲染一个多智能体的迷宫的图。

3. 把lbf环境接入该项目自带的env库，让env = gym_make('Foraging-8x8-2p-1f-v2')可以运行并生成吃水果图。
    * [commit][finished] 在envs中增加lbforaging文件夹和调用全局register.py的环境特定register.py，在envs.__init__中增加`import lbforaging`。
    * [test][planned] 测试env = gym_make('Foraging-8x8-2p-1f-v2')可以运行并生成吃水果图。

可能创新点：
* 现有的adversary环境编写都是依赖于具体环境的，例如minigrid，minihack的Adversary环境分别继承环境具体的类。是否可像OpenAI gym一样编写一个统一Adversarial Env的API，将adversarial Env也引入注册制度。（其实已经被写完了，只是没有继承gym.Env）
* 以minigrid为例，adversary环境的生成过程是多次执行动作，并依赖于时间步的，这里存在用Transformer提高性能的可能性。但目前只看了minigrid一个环境，其他环境还有待研究。可能transformer在大型环境好用，UED在小型环境好用。

经验：
* 如何处理一个原本已经有成熟文档的很大的库，例如gym和torch？目前看法是读一遍文档。注意文档和库版本对应(gym)。文档记录了该库一般设计方法(gym.Env和gym.Wrapper)和常用类的使用方法，这些普遍性的知识最好记住，但可以通过记笔记辅助。对于文档没记录但代码中有的，或者当前项目使用该库的方法的，则需要通过笔记详细记录其特殊性。Python的普遍性一般记到内置类型（元组字典列表）和库的基本函数为止（如numpy和torch的add等）。这些深度没有明确的标准限制，旨在将需要使用的那层学得得心应手即可。如何处理一个原本没有文档，又很大的库？目前看法是读一遍代码，自己写一份文档。如何处理小的库？也直接看代码，可以重构的话就重构。

注：
* 数字标号1，2表示对项目重要部分（例如Env，Agent等）的修改目标，下面的黑点标号表示每一步工作。
* 工作条目中的[commit]表示该条目是git中的一条commit，[test]表示该条目是一次测试，每次commit实现的功能都必须有测试，[doing][planning][finished]分别表示进行中，计划中，已完成。问题条目中的[solved]表示已解决，[unsolved]表示未解决。
* 下面的文档不能只用中文记每个类有什么作用，应该主要记每个类有哪些属性和方法，中文是附属内容，解释这些类的功能。


# PAIRED项目阅读笔记
## gym库结构
### Env(object)
Env是整个Gym库范围内的终极基类。可以说一个强化学习环境除了少量对象（Error，Space，EnvRegistry等），都是Env的子类。

Env类有以下成员：
* metadata：字典，表示render_mode，framerate等元数据。每个继承gym.Env的子类都需要给出metadata。
* reward_range：元组，表示reward的范围。
* action_space, observation_space：表示动作，观测空间。
* window：渲染使用，表示窗口。

有以下方法：
* step：参数为动作，让环境前进一步，返回值为元组（观测值，奖励，终止，截断，信息）
* reset：可选参数为初始化环境的种子，返回值为初始观测值。
* render：渲染环境。根据rendor_mode可以选择渲染方式。这三个方法必须子类实现。
* close：垃圾回收或程序结束时调用的方法。子类可以选择是否重写。

注：Python面向对象继承object类可以获得一些通用方法。

### Wrapper(Env)
一种特殊的Env类，可以包裹另一个Env对象env，并在不改变env代码和API的前提下对step和reset方法进行转换。它应用了设计模式中的代理模式。

有以下成员：
* env：被包裹的Env类实例。
* 所有gym.Env类有的成员变量。

有以下方法：
* 所有gym.Env类有的方法。

### ObservationWrapper(Wrapper)，RewardWrapper(Wrapper)，ActionWrapper(Wrapper)
ObservationWrapper调用observation方法处理step和reset生成的观测，RewardWrapper调用reward方法处理step生成的奖励，ActionWrapper调用action方法处理step输入的动作。这三个方法均需要子类重写。

### TimeLimit(Wrapper)
给env对象增加运行时间步上限的类。

### TransformObservation(ObservationWrapper)，TransformReward(RewardWrapper)
转换观测、奖励的类。

### ClipAction(ActionWrapper)
将连续动作剪切至合法范围内的类。

### Space(object)
用于定义动作和状态空间的基类。明确定义了与环境交互的方法，包括动作和观测的格式。不同的Space可以被容器Space结构化地组合起来，从而构成更具表达性的Space。这使用了设计模式中的组合模式。

有以下属性：
* shape：多个Space组成的形状。
* dtype：Space数据类型。

有以下方法：
* sample：参数mask，随机生成空间中的一个元素，mask表示哪些元素可以被生成。
* contains：参数x，判断x是否为空间中的一个合法元素。
* seed：参数seed，指定随机数种子。

### Box(Space)
$R^n$中一个可能是无界的区间，进一步地，表示n个闭区间的笛卡尔积，每个闭区间可能是$[a,b]$，$(-\infty,b]$，$[a,+\infty)$，$(-\infty,+\infty)$中的一个。可以每一维拥有相同的界，也可以每一维拥有不同的界。

有以下属性：
* low, high：上界和下界。

### Discrete(Space)
包含有限多个元素的空间，进一步说是$\{a,a+1,\cdots,a+n-1\}$。

有以下属性：
* n：表示元素个数。

### MultiDiscrete(Space)
表示多个Discrete Space的笛卡尔积。常用来表示键盘，游戏手柄等。

有以下属性：
* nvec：参数 nvec 将确定每个分类变量可以取值的数量。
* dtype：数据类型，应为整型。

### Dict(Space)
Space实例的字典容器。采样后得到一个相同结构和键，对每个Space对象递归采样的字典。

有以下属性:
* spaces：一个OrderedDict，每个值都必须是gym.Space的实例。

### Tuple(Space)
Space实例的元组容器。采样后得到一个每个元素是对每个Space对象递归采样的元组。

有以下属性：
* spaces：一个tuple，每个值都必须是gym.Space的实例。

### EnvSpec(object)
特定环境实例的参数细节。有init和make两个方法。make根据参数造出一个环境实例。比较重要的属性是entry_point和id。

### EnvRegistry(object)

环境构造器，单例模式，全局仅一个实例，它的每个方法也都包装成静态的。

### batch_space函数
生成一个batched空间，包含单个空间的多个拷贝。

有下列参数：
* space：gym.spaces.Space实例，向量环境中单个环境的Space。
* n：向量环境中环境的个数。

### VectorEnv(gym.Env)
向量化环境的基类。向量化环境的每个观测是每个子环境观测构成的batch。step也期望为每个子环境收到一个动作batch。

有下列成员：
* num_envs：环境个数
* observation_space：观测空间，调用batch_space将单个环境的空间构造为向量化环境的空间。
* action_space：动作空间，直接将单个动作空间放入Tuple(Space)中，构造新空间。
* single_observation_space：单个环境的观测空间。
* single_action_space：单个环境的动作空间。
* closed：是否关闭。
* viewer：渲染有关。

有下列方法：
* reset_async，reset_wait：未实现，异步reset。
* reset：同步reset。调用上面2个方法。
* step_async，step_wait：未实现，异步step。
* step：同步step。调用上面2个方法。参数动作和返回值观测为space采样格式。rewards和dones为np.ndarray格式。
* close_extras：关闭环境，未实现。
* seed：指定随机数种子。

### SyncVectorEnv
维护一个环境的list，单线程依序运行list中每个环境的函数。
用不上，暂时不看。

### AsyncVectorEnv
维护多个并行环境，使用管道通信。
用不上，暂时不看。

## torch库结构

### torch包
torch包含多维张量的数据结构和这些张量上做的数学操作，还提供一些有效序列化的小工具，以及其他有用的工具。

一些全局方法如下：
* is_tensor：判断一个object是否是tensor。
（以下为创建方法）
* tensor：复制参数data，制造一个没有autograd历史的Tensor（也称叶子Tensor）。有无autograd可通过参数required_grads指定。
* zeros：构造一个指定size的全0张量。
* ones：构造一个指定size的全1张量。
* full：构造一个指定size，值全为fill_value的张量。

（以上三个方法均有对应的xxx_like方法。shape由输入张量指定）
* eye：构造一个指定2-D shape的单位阵。

（以下为索引，切片，连接，变异方法）
* cat，concat，concatenate：在给定维度dim拼接给定张量序列。
* gather, scatter：说不清，看源文档示例。
* index_select：按照index和维度dim选择数据。
* masked_select：按照masked和维度dim选择数据。
* split：按照size和维度dim划分张量，得到张量元组。
* squeeze：删除给定维度dim列表中所有size是1的维度。
* transpose：将dim0和dim1两维度转置。
* unsqueeze：在给定维度处增加size是1的维度。
* where：给定一个布尔张量condition和两个同大小的张量input，other，所有condition是True的位置选择input对应位置元素，否则选择other对应位置元素。
（以下为序列化方法）
* save：将张量用pickle序列化，存入磁盘。
* load：从磁盘上读取序列化后的张量。
（以下为梯度计算相关方法，context manager）
* no_grad：禁止梯度计算。
* grad：启动梯度计算。
（以下为数学运算方法，太多了，暂不介绍）
* ...
* ...（其他不太使用，暂不介绍）

### Tensor
张量，支持求梯度操作。

有以下方法：
* size：返回张量的size，可以选择是否指定维度dim。
* ...（Tensor类其他的方法，都在torch内有全局版本，放在上面介绍）

### torch.nn包
建造计算图的基本块。

### Parameter(torch.Tensor)
一种可以被作为参数的张量。有一个特殊性质——在被赋值为module的一个属性的同时，会自动加入参数列表中（通过__setattr__实现），并且出现在parameters()迭代器内。普通的Tensor不会有这种效果。一些情况下可能需要将模型的某些隐藏状态暂时存储起来，例如RNN的上个隐藏状态。这种情况下这些隐藏状态不能被register，所以需要区分可求梯度的parameter和暂时存储的buffer。

### Module
所有模型的基类。任何模型都应该是该类的子类。Modules可以包含其他Modules，构成一个树形结构。可以将子模型分配为一般属性。

有下面这些属性：
* tranning：表示模型是训练模式还是评估模式。

有下面这些方法：
* forward()：抽象方法，模型的计算过程，需要每个子类重写。调用的时候，应该直接调用类本身，而不是调用forward方法。前者关注了注册的hook，但后者忽视它们。
* register_buffer()：插入一个非参数，但又属于模型状态的量，以及对应名字，例如BatchNorm的running_mean。
* register_parameter()：插入一个参数，以及对应名字。参数可以作为一个属性被访问。
* register_backward_hook()：插入
* add_module()：插入一个模型，以及对应名字。
* parameters()：返回可迭代的参数列表
* buffers()：返回可迭代的缓存列表
* children()：返回可迭代的孩子module列表
* modules()：返回可迭代的module列表
* apply()：将某个函数应用到当前module以及所有孩子(children()的返回值)
* cpu()，cuda()：将模型的所有parameter，所有buffer移动到对应设备，并应用于所有子模型。
* type()，float()，double()，half()等：将模型的所有parameter，所有buffer转变为对应类型，并应用于所有子模型。
* to()：支持kwargs，以该形式调用上面所有的函数对module做出变换，并应用于所有子模型。
* state_dict()：返回一个OrderedDict，表示模型所有状态
* load_state_dict()：读取一个Dict，并修改模型所有状态
* train()，eval()：对当前module和所有子module修改Training
* require_grad()，zero_grad()：对当前module的所有parameter修改require_grad属性。

### ModuleList
用一个List维护模型，可以用下标访问。与Python的List比，其内部所有模型均会在加入时被注册为子模型，并被所有Module的函数识别到。

有下面这些方法：
* append：在末尾添加一个模型。
* insert：在给定下标index之前添加一个模型。

### Sequential
顺序容器。模型会按照传入构造函数的顺序被加入容器中，也可以传入一个OrderedDict。模型的forward方法接受输入，将其前向传入容器内第一个模型，然后按链式顺序将每个输出都传入下一个输入，最后返回最后一个模型的输出。

有下面这些方法：
* append：在末尾添加一个模型。
### ......很多具体的Module，暂不介绍

### autograd包
autograd是一个反向自动求导系统。概念上，autograd记录一个包含所有生成数据操作的有向无环图，其叶子是输入张量，根是输出张量。从根到叶子追踪该图，就可以用链式法则自动计算梯度。

在内部，autograd通过一个Function对象的图来表示。在前向pass计算时，autograd同时执行要求的计算，并建立一个图表示计算梯度的函数，每个torch.Tensor的grad_fn属性是该图的一个进入点（entry point）。前向pass计算完成后，我们通过后向pass评估该图来计算梯度。

注意，该图在每次循环时都从头开始构建。

一些操作需要暂时存储前向pass的中间结果，用于执行反向pass。例如计算$f(x)=x^2$的梯度

### Function
用来生成可autograd函数的基类。为了构造一个一般的autograd函数，需要继承该类，并实现forward和backword静态方法。在前向pass中使用该函数时，调用apply方法，而不是直接调用forward方法。

有下面这些方法：
* forward：前向计算结果，并用ctx为backward保存临时变量。
* backward：反向计算梯度，使用ctx中的变量。


## 参数(make_cmd.py)
参数是项目必须的一部分，包含环境参数、模型参数、学习参数、日志参数。

该项目指定参数有四种方法：
* 直接在代码中指定默认参数，得到字典类型结果。
* 使用json格式的配置文件指定参数，可在运行时使用仓库读取，得到字典或列表类型的结果（有些项目使用yaml）。
* 使用命令行参数并通过argparse包的ArgumentParser类解析，得到Namespace类型结果。
* 在使用add_argument方法配置ArgumentParser包时指定默认参数，也得到Namespace类型结果。
若后两者同时使用，则其结果位于同一个Namespace实例中。

在运行核心功能之前，需要将这些参数合并到一起，以字典、列表类型结果为最终结果（Python内置类型）。

如有需要，可以让参数处理模块单独成为一个文件，将不同方式指定的参数转换为完整的命令行。最后运行命令行，统一使用ArgumentParser包读取参数后进行处理。

## 日志(FileWriter类)
首先指定一个所有日志存放的根目录，例如~/logs/paired。然后让一个参数(--xpid)的值是这次运行代码时，所有日志文件存放的目录。例如--xpid的参数值是train，则所有日志文件均被存放在~/logs/paired/train内。实际工程实践中，--xpid的参数值经常使用当前时间对应的字符串或者当前进程的pid。可用time包的strftime函数生成字符串。time包中维护的time类型是struct_time类型。

日志内容可以使用python的logging库来输出，如果logging库不能满足要求，还可以使用该库自己封装一个类，如FileWriter。

## 训练(train.py)
训练代码主要分以下几个部分：

首先是初始化部分：
* 使用os.environ设置环境变量。
* 使用ArgumentParser类解析参数。
* 日志初始化。包括配置截图目录、日志文件目录、模型目录等。以及对logging包，baseline的logging包进行配置。log_stats函数在命令行输出信息。
* 硬件初始化。判断是否有GPU，是否安装CUDA和CuDNN，以及torch和tf是否使用了它们。
* 训练环境venv初始化。构造并行的训练环境对象。使用create_parallel_env函数，以及作者自己写的一套并行环境库envs，为每个环境配一个并行版本。
* 智能体初始化。构造参与训练的智能体。使用make_agent函数，以及作者自己写的一套包括agent的库algo。
* signal处理器初始化。处理非同步进程的信号，关闭环境。使用python的signal库的signal函数。
* runner初始化。训练模型的类，参数包括智能体们和并行环境们。有一个表示状态的stat_dict成员。有训练、评估方法以及读写state_dict的方法。
* checkpoint读取。包含模型的checkpoint和训练过程的checkpoint。代码中都是用torch的load方法来读取二者？可能训练过程的dict也放在模型里面了。
* evaluator初始化。评估模型的类，参数包括环境名，进程数，回合数，设备。

其次是训练部分：
* 用timeit库的timeit方法为代码运行计时。
* 进入训练大循环。其中每一个循环需要做的事情如下：
    - 调用一次runner.run，完成一次训练。
    - 判断本轮是否需要记录日志，是否需要截屏。
    - 如果需要记录日志，则先判断是否需要评估，并将评估结果存入stats中，然后进行日志记录和状态更新。
    - 判断是否需要保存checkpoint或者备份checkpoint，如果是，则执行。
    - 如果需要截屏，则保存截屏。
    - 关闭evaluator和venv。


## create_parallel_env函数
该函数主要分以下几个步骤：
* 获取一个环境生成函数，从args中处理参数，并获取从gym改写而来的EnvRegistry类的make函数，作为lambda表达式。
* 使用ParallelAdversarialVecEnv类，获取一组并行的adversarial环境venv。
* 将生成的venv套上一些wrapper，得到新的venv和ued_venv。

该函数主要作用是生成“环境”。具体是
* 使用自主编写的环境库envs中的环境工厂EnvRegistry产生简单环境实例。
* 维护并行环境容器。其最顶层基类为VecEnv类，其他类是继承自VecEnv的有具体实现的并行子类(SubprocEnv类和ParallelAdversarialVecEnv类)或套在VecEnv实例上用于升级功能的wrapper类的。

这些类介绍如下：

### VecEnv
能维护一组异步、向量化的同一个环境实例的抽象类，是PAIRED工程中所有环境相关类的父类。用于从环境的多个复制对象中获得，所以观测是batch观测，期望的动作也是batch型动作，每个环境执行一个动作。

该类有下列属性：
* metadata：同gym.Env
* num_envs：并行环境个数。
* observation_space：并行环境的观测空间。
* action_space：并行环境的动作空间。

该类有四个需要子类实现的抽象方法：
* reset：对每个环境实例进行reset，并返回一组观测值。若step_async仍在运行，则会停止运行。step_wait在下次step_async调用之前都不能调用。
* step_async：通知环境实例（但不一定是所有实例），开始根据给定action执行自己的step。调用step_wait收集结果。
* step_wait：等待所有step完成，并返回(obs, rews, dones, infos)。
* get_image：获得渲染。

除了reset还有三个对外提供的gym API方法：
* step：先调用step_async，再调用step_wait，返回多个环境的(obs, rews, dones, infos)。
* render：实现渲染。调用一些内部方法，如get_images和get_viewer等。
* close：关闭环境，包括viewer等资源。

### VecEnvWrapper(VecEnv)
Proxy设计模式。包含一个venv（谁的实例？SubprocEnv），保证venv的API不变，并在此基础上修改功能。

有以下属性：
* venv：一个并行环境实例，经常为SubprocEnv的实例。
* num_envs：并行环境个数。
* observation_space：并行环境的观测空间。
* action_space：并行环境的动作空间。
（上面三个和VecEnv一样）

有以下方法：
* step_async，close，render，get_images：直接调用venv的对应方法。
* reset, step_wait：继续抽象方法。

### SubprocEnv(VecEnv)
通过python的subprocess实现多个环境并行，通过管道Pipe实现通信的一种VecEnv实现。

有以下属性：
* observation_space，action_space：与env_func生成环境相同。

构造函数：
* 按环境数nenvs和每组环境数in_series划分进程数。
* 创建一组管道，得到（SubprocEnv侧,环境实例侧）管道端元组的列表。用zip可以将所有SubprocEnv侧管道端和环境实例侧管道端分别聚合在一起。
* 创建进程实例列表。所有子进程都运行一个worker函数，worker函数会接受管道发去的命令和参数，将这些参数在它维护的几个环境里都运行一遍。

另外还实现了step_async，step_wait，reset等方法。这些方法均通过管道发命令到子进程，再接受子进程的命令结果。

### ParallelAdversarialVecEnv(SubprocEnv)
支持普通环境，也支持adversarial环境的并行环境容器。

有以下属性（除了action_space以外都用__getattr__指定）：
* adversary_action_space：敌对环境的action_space。
* adversary_observation_space：敌对环境的observation_space。
* observation_space：普通环境的observation_space。
* action_space：用VecEnv的。

实现了seed，level_seed以及对应的async和wait方法，以及step_adversary_async和step_adversaty（其中wait使用SubprocEnv的）

### VecMonitor(VecEnvWrapper)
收集环境信息，存到step的info里。

### VecNormalize(VecEnvWrapper)
对环境观测值和奖励归一化。

### VecPreprocessImageWrapper(VecEnvWrapper)
似乎是用来截图的。把所有的observation转变为图片形式。

### AdversarialObservationWrapper(gym.core.Wrapper)
该类为Environment Generating Adversarial的API代理。不知道为什么这个又去继承Env类（猜想原因是作者自己实现了VecEnv，没有去用gym的VecEnv）。

### MultiGridFullyObsWrapper(AdversarialObservationWrapper)
完成观测MultiGrid环境的Wrapper。不知道为什么这里要单独写一个Wrapper。

## Minigrid环境结构
gym_minigrid包包含一个envs包和一些单独的py文件。
初始化包即初始化内部的envs包，并引入wrappers.py中为这个环境准备的wrapper们（继承自gym.core.wrapper系列）。

### envs包
初始化：引入该包内所有环境py文件的所有类。这些类本身是由另一套类构成的。

### WorldObj
格子世界中所有对象的基类。

有下面这些属性：
* type：对象类别。必须为'unseen','empty','wall','floor','door'.'key','ball','box','goal','lava','agent'其中之一。
* color：对象颜色。必须为'red','green','blue','purple','yellow','grey'其中之一。
* contains：包含的对象。一般为None，但盒子可以包含其他对象，这时则不是None。
* init_pos：对象初始位置。
* cur_pos：对象当前位置。

有下面这些方法：
* can_overlap：是否能被覆盖。默认False。
* can_pickup：是否能被捡起来。默认False。
* can_contain：是否能包含其他对象。默认False。
* see_behind：是否能看到该对象后方。默认True。
* toggle：触发一个Object执行的动作的方法。默认False。（？？）
* encode：用三个整数的元组编码这个对象。
* decode：根据三个整数的元组解码一个对象实例，解码出的实例是各个对象的实例，不是WorldObj的实例。
* render：渲染物品的方法。子类实现。

### Goal(WorldObj)
* type：goal
* can_overlap：True。
* color：green。
* 其余同WorldObj。

### Floor(WorldObj)
* type：floor
* can_overlap：True。
* color：blue。
* 其余同WorldObj。

### Wall(WorldObj)
* type：wall
* see_behind：True。
* color：grey。
* 其余同WorldObj。

### Door(WorldObj)
* type：door
* is_open, is_locked：默认锁上
* can_overlap, see_behind：和是否锁上有关。
* color：参数指定。
* toggle：判断是否带了对应钥匙，锁上但没带钥匙则False。
* 其余同WorldObj。

还有Key, Ball, Box(WordObj)
各种游戏元素，分别实现自己的render方法，以及和默认值不同的属性。

### Grid
地图类，表示格子世界和上面的操作。

有下面这些属性：
* width, height：地图宽和高。
* grid：整个地图的元素，是一个长为width*height的list。

有下面这些方法：
* set，get：修改，读取某个位置的元素。
* horz_wall, vert_wall：画一排墙。
* wall_rect：画四排墙组成矩形围城。
* rotate_left：地图逆时针转90°
* slice：获取grid的一个子集。
* render_tile，render：渲染，暂时不看。
* encode，decode：将Grid地图与（width,height,channel）的numpy数组互相转换。
* process_vis：不知道干嘛，用到了再看。

### MiniGridEnv(gym.Env)
MiniGrid世界的gym.Env环境类。

有下列属性：
* action_space：Discrete(7)
* agent_view_size：智能体可以看见的范围。
* observation_space：一个Dict，'image': shape为(agent_view_size,agent_view_size,3)的[0,255]整型Box。
* reward_range：(0,1)
* width,height：地图宽高。
* max_steps：最大步数。
* see_through_walls：是否能看到墙后面。
* agent_pos和agent_dir：智能体当前位置和方向。
* carry：智能体拿着的对象，一般为None。

有下列方法：
* reset：重新生成width*height的地图，智能体位置和方向，清空max_steps和carrying。调用gen_obs返回obs = {
            'image': image,
            'direction': self.agent_dir,
            'mission': self.mission
        }。注意Dict只影响Sample。
* step：执行动作。调用gen_obs返回obs。
* render：渲染，暂时不看。
* gen_obs：生成上面的字典obs。
* put_obj：将某对象obj放在某位置(i,j)。
* place_agent：设置agent_pos和agent_dir。通过调用place_obj放置None来实现。注意agent位置通过agent_pos和agent_dir实现。
* place_obj：尝试将某对象obj放在某范围内随机一个位置。
等一堆为了实现上面三个方法而单独写的方法。
* _gen_grid：需要子类实现的方法，初始时随机生成各种不同主题的地图。

### Room
四周被墙包围的房间，每条边上最多一扇门（或者地板）。

有下列属性：
* top,size：房间左上角位置和大小。
* doors：四面门或地板对象。
* door_pos：四面放门或地板的位置。
* neighbor：相邻房间。
* locked：是否锁住。
* objs：包含的对象。

### RoomGrid(MiniGridEnv)
由多个房间和随机对象组成的环境。

有如下属性：
* Env系属性基本同父类
* room_size：房间大小
* num_rows, num_cols：房间个数长宽。
* 其他MiniGrid系属性基本同父类。

### EmptyEnv(MiniGridEnv)
空环境。四周是墙，中间只有智能体和终点。

有以下参数：
* agent_start_pos，agent_start_dir：可选，指定智能体初始位置。

有以下方法：
* _gen_grid：空地图，wall_rect四周放围墙，put_obj放终点，agent_start_pos，agent_start_dir不存在则使用place_agent放智能体。

### EmptyEnv5x5(EmptyEnv)，EmptyRandomEnv5x5(EmptyEnv)，EmptyEnv6x6(EmptyEnv)，EmptyEnv6x6(EmptyEnv)，EmptyEnv16x16(EmptyEnv)
生成一些给定不同参数的空环境类。并在empty.py被import时通过调用gym.envs.register来注册。

### ???Env
其他的Env们和EmptyEnv一样的思路。先通过继承MiniGrid，构造特殊的Grid类型地图。有些为了方便，先继承RoomGrid，构造出Room之后再单独修改。这些到后面再研究。

### Window
使用matplotlib渲染出格子世界的类。可以用并行版本的环境实例和Wrapper生成图片。

## Multigrid环境结构
该环境实现了MultiGrid类和WorldObj类，分别继承MiniGrid类和minigrid.WorldObj类。

### WorldObj(minigrid.WorldObj)
WorldObj类的区别是decode时多加了一个Agent类，将Agent类也作为Obj的一种。WorldObj是多智能体环境的所有对象父类。

### Door(WorldObj)
没看懂为什么要特地继承一个Door出来。说是能适合多智能体环境。

### Agent(WorldObj)
继承一个Agent，表示Agent也是一种Obj。这里比较重要，取消了Minigrid单放Agent的特性。

### Grid(minigrid.Grid)
Grid也继续继承，得到适合多智能体场景的Grid。具体改的有render时会注意枚举多个智能体，并渲染。decode时把原本用三角形表示智能体的代码去掉，统一为调用agent类。

### MultiGridEnv(minigrid.MiniGridEnv)
MultiGridEnv继承MiniGridEnv，包含gym.Env系属性和MinigridEnv属性，还增加了多智能体支持。

有下列属性：
* n_agents：表示智能体个数。
* competitive：是否仅一个智能体能到终点。
* agent_view_size：智能体看到的路径。
* reward_range：奖励范围(0,1)
* direction_obs_space：智能体方向的观测空间。整型[0,3] n_agents维Box。
* minigrid_mode：是否为单智能体，结果影响后面两个属性。
* action_space：minigrid_mode为True则是Discrete Space，否则是长度n_agents的Box Space。
* image_obs_space：minigrid_mode为True则是三维图像Box Space，否则是多一维n_agents的四维Box Space。
* observation_space：由{'image': self.image_obs_space,
         'direction': self.direction_obs_space}构成的Dict Space。
* width, height：环境大小
* max_steps：最大步数
* see_through_walls：能否看到墙后面。
* agent_pos：列表，每个智能体的位置。
* agent_dir：列表，每个智能体的方向。
* done：每个智能体是否完成任务。
* fixed_environment：是否在每次reset环境的时候使用相同的随机数种子，即是否每次reset环境时都是相同的环境。

有下列方法：
* \_\_init\_\_：没有调用minigrid的构造函数，重新对minigrid存在的属性做了赋值。
* reset：根据fixed_environment初始化种子，初始化agent_pos,agent_dir和done。调用_gen_grid生成地图。检查是否有智能体覆盖其他Obj，初始化carrying。
* _gen_grid：生成地图，包含所有agent的位置，wall的位置，Goal的位置，以及其他位置。
* place_obj：随机在地图上找一个可放置Obj的位置，不能与其他Obj和智能体重合。若放置智能体，则需调用下面三个方法再完成放置；放置其他对象可以直接放置。
* place_agent，place_one_agent, place_agent_at_pos：放置智能体，通过调用place_obj并以None为参数，找到智能体可以放的位置，再放置，并更改各种属性。
* step以及实现它的各种小方法：多智能体step之前会随机打乱顺序。
* render以及实现它的各种小方法。

### MazeEnv(multigrid.MultiGridEnv)
简单的单智能体环境，可以看到墙后面。

有以下属性：
* start_pos，goal_pos：智能体和目标的位置。
* bit_map：表示地图数据的二进制数组。0表示空地，1表示墙。

### HorizontalMazeEnv(MazeEnv)，Maze3Env(MazeEnv)，SmallCorridorEnv(MazeEnv)，LargeCorridorEnv(MazeEnv)，LabyrinthEnv(MazeEnv)，Labyrinth2Env(MazeEnv)，NineRoomsEnv(MazeEnv)，NineRoomsFewerDoorsEnv(MazeEnv)，SixteenRoomsEnv(MazeEnv)，SixteenRoomsFewerDoorsEnv(MazeEnv)，MiniMazeEnv(MazeEnv)，MediumMazeEnv(MazeEnv)
各种继承自MazeEnv，并通过指定start_pos，goal_pos和bit_map构建的迷宫。在文件末尾还选择了一些注册到registry（不是gym里那个，是envs/registration里实现的）。

### FourRoomsEnv(MiniGridEnv)
继承MiniGridEnv，支持指定目标和智能体位置，自带四个房间的环境。

### CrossingEnv(MiniGridEnv)
继承MiniGridEnv，带有河流和岩浆的环境。

### LavaCrossingEnv(CrossingEnv)，LavaCrossingS9N2Env(CrossingEnv)，LavaCrossingS9N3Env(CrossingEnv)，LavaCrossingS11N5Env(CrossingEnv)，SimpleCrossingEnv(CrossingEnv)，SimpleCrossingS9N2Env(CrossingEnv)，SimpleCrossingS9N3Env(CrossingEnv)，SimpleCrossingS11N5Env(CrossingEnv)
不同参数的特定CrossingEnv，全部注册到registry（不是gym里那个，是envs/registration里实现的）

### AdversarialEnv(multigrid.MultiGridEnv)
对手建造的让agent参与交互的环境。建造方法是按顺序放置目标，智能体和不超过n_clutter个障碍物。有gym.Env,MinigridEnv,MultigridEnv三个类的属性。

有下列自身的属性：
* agent_start_pos：智能体初始位置，默认None。为什么不是多智能体？
* goal_pos：目标位置，默认None。
* n_clutter：最多放置的障碍数。
* goal_noise：目标随机离开对手选择的位置。
* random_z_dim：环境生成一个随机向量z表示对手，这是该向量的维度。
* choose_goal_last：如果是True，则先放墙，后放智能体和目标。
* adversary_max_steps：对手最多步数，障碍数+2。
* distance_to_goal：
* n_clutter_placed：
* deliberate_agent_placement：
* passable：
* shortest_path_length：
（这5个是后期实验的测量标准）
* adversary_action_dim：地图除了四周围墙剩下的面积，对手动作维度。
* adversary_action_space：对手动作空间，adversary_action_dim维的Discrete Space。
* adversary_ts_obs_space：time_step观测空间，0~adversary_max_steps的1维uint型Box。
* adversary_randomz_obs_space：随机生成的向量空间，0~1的random_z_dim维float32型Box。
* adversary_image_obs_space：图像形式的观测空间，0~255的(self.width, self.height, 3)维度的uint8型Box。
* adversary_observation_space：由{'image': self.adversary_image_obs_space,
         'time_step': self.adversary_ts_obs_space,
         'random_z': self.adversary_randomz_obs_space}构成的Dict。
* graph：NetworkX库的格子图，可以求最短路径。
* step_count，adversary_step_count：智能体和对手执行步数。
* agent_start_dir：所有智能体开始方向。
* agent_pos，agent_dir，done，carrying：minigrid就有，multigrid变成list。

有下列方法：
* _gen_grid：直接生成四周的墙。
* reset_agent_status：重置agent_pos，agent_dir，done，carrying。
* reset_agent：重置agent的开始位置，但不重置目标和墙。
* reset：重置格子图。
* step_adversary：给定一个整数loc（adversary_env的动作）表示位置，根据choose_goal_last决定一系列动作中，墙、智能体、目标分别是哪几步。
* reset_random：随机放置Obj（是什么Obj和时间步有关）。一般用于domain randomization。

### ReparameterizedAdversarialEnv(AdversarialEnv)
格子世界，支持adversary建造新环境给智能体参与。
Adversary对每个格子都执行一次动作，每个格子都有四种动作：放智能体，放目标，放墙，什么也不做。如果某次放智能体和目标放在了与之前不同的地方，则它们会按照新位置算。

### MiniAdversarialEnv(AdversarialEnv)，MiniReparameterizedAdversarialEnv(ReparameterizedAdversarialEnv)，NoisyAdversarialEnv(AdversarialEnv)，MediumAdversarialEnv(AdversarialEnv)，GoalLastAdversarialEnv(AdversarialEnv)，GoalLastOpaqueWallsAdversarialEnv(AdversarialEnv)，GoalLastFewerBlocksAdversarialEnv(AdversarialEnv)，GoalLastFewerBlocksOpaqueWallsAdversarialEnv(AdversarialEnv)，MiniGoalLastAdversarialEnv(AdversarialEnv)，FixedAdversarialEnv(AdversarialEnv)，EmptyMiniFixedAdversarialEnv(AdversarialEnv)
一系列具有参数的，支持adversary选择动作，建造迷宫的新环境实例。

### 本项目的环境Register流程（不是gym库）
envs/registration.py是实现registry类，并实例化一个全局registry，提供全局的make，register和spec函数用于环境注册。
每个环境包内的register.py是复制的gym，提供一个特定的register函数，专门完成该环境调用全局register的工作。调用这个特定register函数在每个环境包的adversarial.py的最后。整个envs包初始化（即__init__.py）时通过import adversarial 来执行这些register函数，其中重要的参数是id和entry point。

调用make时，会根据注册时传入的spec里的entry_point，来对接实际环境的底层代码，得到一个真正环境的实例。将这些实例放进ParallelAdversarialVecEnv里，再套上一些继承自VecEnv的Wrapper即可得到最后的环境。

总之，envs包初始化时，会通过所有环境自身的register函数去调用registry的register函数传入spec，从而完成所有环境的注册。调用make函数时，根据给定的id，找到对应spec的entry_point，来对接实际环境的底层代码，最后得到环境的实例。并行环境类可以多次调用make，获得并行环境，并且套上一些Wrapper，最后完成并行环境构造工作。

## make_agent函数

该函数主要分以下几个步骤：
* 根据args，之前生成的并行环境实例（继承自VecEnv，有可能套Wrapper）以及智能体种类（Adversary, Protagonist, Antagonist三者之一），生成后续参数给ppo算法，storage，ACAgent三者使用。
* 根据训练类型配置，调用model_for_env_agent函数。该函数将传入的venv也作为参数，并根据环境类别再分别调用model_for_multigrid_agent和model_for_minihack_agent生成智能体模型（MultiGridNetwork）。
* 通过参数们构造PPO算法和RolloutStorage，最终构造ACAgent并返回。

该函数主要作用是生成“智能体”，即agent，adversary_agent和adversary_env其中之一。根据不同的env环境参数，智能体分别为MultigridNetwork，MiniHackAdversaryNetwork等类型。这些类型最终均继承自torch.nn，以env环境的各种参数（如observation_space，action_space等）为参数构建神经网络。

### model_for_env_agent函数
输入参数：环境名env_name，并行环境实例env（继承自VecEnv，有可能套Wrapper），智能体类型agent_type（'agent'，'adversary_agent'和'adversary_env'之一），剩余参数可以去arguments.py中找。

分别根据环境名，调用相关的model_for_xxx_agent函数获取智能体模型。

### model_for_multigrid_agent函数
输入参数：并行环境实例env（继承自VecEnv，有可能套Wrapper），智能体类型agent_type（'agent'，'adversary_agent'和'adversary_env'之一），RNN内部结构recurrent_arch（'lstm'，'gru'之一），RNN的隐藏层维度recurrent_hidden_size（这个不是很懂是什么）。

根据agent_type，以不同参数实例化MultigridNetwork类并返回。

### DeviceAwareModule(nn.Module)
可通过.device获取自身所在设备的模型类。

### MultigridNetwork(DeviceAwareModule)
面对MiniGrid环境使用的网络模型。输入为图像，中间有一些RNN隐藏层，输出一侧有Policy和Value两个头。

属性如下：
* num_actions：每个动作有多少种选择。
* multi_dim：是否有多个动作。
* action_dim：动作的个数。
* num_action_logits：不知道是干嘛的，根本没有用上，注释掉了。

### PPO
执行PPO算法的类。参数有模型actor_critic，PPO算法的超参clip_param, ppo_epoch, lr, eps等。
有__init__和update两个方法。__init__负责初始化超参。update负责从rollout中获取数据，并使用这些数据更新模型参数。

### RolloutStorage
用于存储智能体在环境中获得的数据的类。参数有模型model，步数num_steps，进程数num_processes，以及对应环境的observation_space, action_space等。
方法有： to可以将该类所有数据转移设备。get_obs和copy_obs_to_index可以根据下标取数据。insert可以插入一条数据。
after_update是配套的PPO的update函数运行结束后对数据做处理的函数。compute_returns计算回报，包括gae和discounted两种回报。
feed_forward_generator和recurrent_generator是数据生成器，可以从这里面取数据，不断进行训练。

### ACAgent
包含PPO和RolloutStorage两个成员，所有方法都是转化到直接调用这两个成员的方法上去。

### Signal Handler这几句
Ctrl+C中断时保证调用venv.close，从而不发生资源泄露问题。

### AdversarialRunner(object)
首次提供对抗性的环境。给定protagonist (agent), antagonist (adversary_agent)和environment adversary (advesary_env)，AdversarialRunner类负责操作这些类执行整个算法流程。该类有上面这三个成员。

五个重要参数定义：
* venv: Vectorized, adversarial gym env with agent-specific wrappers.
* ued_venv: Vectorized, adversarial gym env with adversary-env-specific wrappers.
* agent: Protagonist trainer. 
* adversary_agent: Antogonist trainer.
* adversary_env: Environment adversary trainer. 参数和前两个不一样。
分两个不同类型的env和三个不同类型的agent。

提供reset，train，eval，stat_dict，load_stat_dict，run等方法。
train和eval是让所有智能体类执行各自的train和eval。
stat_dict，load_stat_dict是对状态的处理。
reset为重置环境。
agent_rollout：下面详细介绍
run给外界训练代码调用，表示一次训练流程，其中三个agent各调用一次agent_rollout。

### agent_rollout函数
该函数执行步骤如下：
* 根据参数初始化环境。把初始观测用copy_obs_to_index插入storage中。
* 循环num_steps步，其中每一步先用get_obs方法从storage中获得观测。
* 调用agent的act方法，传入obs，recurrent，hidden_state和mask，返回value, action, action_log_dist, recurrent_hidden_states
* 根据action_log_dist算action_log_prob，

## 其他重要依赖
baseline（不是stable_baseline3）库的logger包内，HumanOutputFormat的writekvs方法可以把stat_dict转换成表格输出。
collections.OrderedDict：支持头尾弹出，将某个元素移动到头尾的字典。

## 设计模式
工厂模式：gym中开一个全局的工厂类，专门用来实例化环境。例子是Registry。

原型模式：将复制对象这件事本身让对象自己执行。只传一些参数表示需要的对象的prototype。例子是这里的EnvSpec。

代理模式：一堆wrapper把环境变成并行+归一化+会写日志+有图像渲染版。
??模式（行为模式里的一个，还没想到）：PPO算法

## 其他
/home/drl/miniconda3/envs/UED/lib/python3.8/site-packages/gym/logger.py:30: UserWarning: WARN: Box bound precision lowered by casting to float32
  warnings.warn(colorize('%s: %s'%('WARN', msg % args), 'yellow'))
表示新建了一个进程，进程里面有一个gym.Env环境实例。

env里的registration.py完全复制gym库的同名文件