# UAV-auto-navigation-and-object-tracking-based-on-RL
毕业设计的代码部分，实现了UE4和airsim环境下无人机自主导航和目标跟踪的强化学习算法。

UE4部分主要参考了以下工程https://github.com/harvard-edge/AirLearning ，算法和接口部分自行实现，因为疫情原因学校进不去没办法去制作docker了，都是copy出来的文件，环境问题可能需要大家自行去解决一下了。

## 1.UAV RL仿真环境调研
关于仿真环境，笔者当时主要针对于开源和free的项目进行。调研了几个开源的仿真环境，并做了对比：

**gazebo**

体积小运行快，视觉效果比较差，物理规则简单，但是是ros原生的仿真器，对于后期RL迁移系统部署会很方便，而且可开发性很强，API主要接口是c++和py。

**unity**

几款流行的物理引擎之一，画质清晰，一些开源项目中是用unity的，但是貌似用的人没有下面的UE4多，所以没有过多调研，放下一些项目链接：

PS：现在unity也有airsim的支持了

mbaske/ml-drone-collection

​https://github.com/mbaske/ml-drone-collection

phachara-laohrenu/DQN-ObstacleAvoidance-FixedWingUAV

https://github.com/phachara-laohrenu/DQN-ObstacleAvoidance-FixedWingUAV

第二个项目很适合做路径规划任务。

**UE4+airsim**

是比较多的大型项目的标配了，也是毕设所采用的主要环境，开发性还好，环境部分可以用ue4c++定制，而且微软airsim的文档维护的也比较好，各种API用着也还可以（有时候也会觉得蛋疼），文档全，photorealistic，适合DRL训练，后期开发性也不错，开源项目很多可以学习和开发，缺点是对平台要求挺高（电脑最好给力点），运行比较慢，项目版本问题有时候会让人很蛋疼，后期切换到实际系统可能会费很大劲。

一些开源项目的链接：

aqeelanwar/PEDRA

https://github.com/aqeelanwar/PEDRA

特别适合导航任务，场景丰富，同时飞机模型也比较多。

airlearning

https://github.com/harvard-edge/airlearning

比较适合DRL的开发场景，集成了env setting，支持domain randomization，对于强化学习算法来说，data的质量应该是非常高的，训练起来个人觉得成功率也会比较大，所以作为了本项目的主要环境进行后续开发，版本为UE4的4.18.3。

## 2.

