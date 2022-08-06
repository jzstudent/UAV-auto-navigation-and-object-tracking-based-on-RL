# UAV-auto-navigation-and-object-tracking-based-on-RL
毕业设计的代码部分，实现了UE4和airsim环境下无人机自主导航和目标跟踪的强化学习算法。

b站对应地址：

https://www.bilibili.com/video/BV1Yr4y1W7iD?vd_source=4650bebdc1f550a1365c594cd887b477#reply122072948160

https://www.bilibili.com/video/BV14R4y1L7MQ/

UE4部分主要参考了以下工程https://github.com/harvard-edge/airlearning-ue4/tree/ffb63b98e1e5e7f9ad1096e3e51b70823c024a6d， 算法和接口部分是自行实现的，这里不建议使用https://github.com/harvard-edge/airlearning-rl/tree/dcd8a39f77992873f49b6872aa534a64a2bfa23a 这个库开发，笔者在此基础上做了很多优化和改进（踩了很多坑）。

PS：因为疫情原因毕业生进不去学校没办法去制作docker了，都是copy出来的文件，环境问题可能需要大家自行去解决一下了。

## 1.UAV RL仿真环境调研
关于仿真环境，笔者当时主要针对于开源和free的项目进行。调研了几个开源的仿真环境，并做了对比，供需要的人使用：

### gazebo

体积小运行快，视觉效果比较差，物理规则简单，但是是ros原生的仿真器，对于后期RL迁移系统部署会很方便，而且可开发性很强，API主要接口是c++和py。

### unity

几款流行的物理引擎之一，画质清晰，一些开源项目中是用unity的，但是貌似用的人没有下面的UE4多，所以没有过多调研，放下一些项目链接：

PS：现在unity也有airsim的支持了

mbaske/ml-drone-collection

​https://github.com/mbaske/ml-drone-collection

phachara-laohrenu/DQN-ObstacleAvoidance-FixedWingUAV

https://github.com/phachara-laohrenu/DQN-ObstacleAvoidance-FixedWingUAV

第二个项目很适合做路径规划任务。

### UE4+airsim

是比较多的大型项目的标配了，也是毕设所采用的主要环境，开发性还好，环境部分可以用ue4c++定制，而且微软airsim的文档维护的也比较好，各种API用着也还可以（有时候也会觉得蛋疼），文档全，photorealistic，适合DRL训练，后期开发性也不错，开源项目很多可以学习和开发，缺点是对平台要求挺高（电脑最好给力点），运行比较慢，项目版本问题有时候会让人很蛋疼，后期切换到实际系统可能会费很大劲。

一些开源项目的链接：

aqeelanwar/PEDRA

https://github.com/aqeelanwar/PEDRA

特别适合导航任务，场景丰富，同时飞机模型也比较多。

airlearning

https://github.com/harvard-edge/airlearning

比较适合DRL的开发场景，集成了env setting，支持domain randomization，对于强化学习算法来说，data的质量应该是非常高的，训练起来个人觉得成功率也会比较大，所以作为了本项目的主要环境进行后续开发，版本为UE4的4.18.3。

## 2.UE4安装
主要是在Ubuntu下配置的

### （1）安装UE4，版本4.18.3：
```
git clone -b 4.18 https://github.com/EpicGames/UnrealEngine.git
cd UnrealEngine 
./Setup.sh 
./GenerateProjectFiles.sh 
make
```
### （2）安装project：
git clone https://github.com/harvard-edge/airlearning-ue4.git

这里要先编辑AirLearning.uproject，删除这个地方

	"Plugins": [
		{
			"Name": "AirSim",
			"Enabled": true
		}
	],
  
然后才可以compile并打开这个项目。

### （3）下载本项目中的airsim-rl
```
  ./setup.sh
  ./build.sh
```
setup.sh这里可能会遇到问题，修改setup.sh下面eigen的地方为：
```
rm -rf ./AirLib/deps/eigen3/Eigen
echo "downloading eigen..."
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.2/eigen-3.3.2.zip
unzip eigen-3.3.2.zip -d temp_eigen
mkdir -p AirLib/deps/eigen3
mv temp_eigen/eigen*/Eigen AirLib/deps/eigen3
rm -rf temp_eigen
rm eigen-3.3.2.zip
```
airsim安装好了之后，将Unreal/Plugins  copy到workspace/airLearning-ue4下即可，接下来编辑AirLearning.uproject为
```
{
	"FileVersion": 3,
	"EngineAssociation": "4.18",
	"Category": "",
	"Description": "",
	"Modules": [
		{
			"Name": "JsonParsing18Version",
			"Type": "Runtime",
			"LoadingPhase": "Default",
			"AdditionalDependencies": [
				"Engine",
        "AirSim"
			]
		}
	],
	"Plugins": [
		{
			"Name": "AirSim",
			"Enabled": true
		}
	],
	"TargetPlatforms": [
		"MacNoEditor",
		"WindowsNoEditor"
	]
}
```
接下来打开工程这一步很重要，不要直接打开project，否则是启动不了的，需要进到UnrealEngine installation folder and start Unreal by running ./Engine/Binaries/Linux/UE4Editor，需要启动UE4Editor，然后点击浏览，选择打开AirLearning.uproject，这个时候会弹出一个框图，选择options中的原位转换，这时才会rebuild相关的文件，等待片刻，就可以了。

### （4）设置Game Mode

好了，到这一步，build+start就可以了，环境配置完成。

## 3.Domain randomization的实现

这部分的代码逻辑主要是上文的https://github.com/harvard-edge/airlearning-rl/tree/dcd8a39f77992873f49b6872aa534a64a2bfa23a 工程中ue4c++和蓝图类配合实现的，通过读取环境配置在每局游戏开始后自动生成生成环境的尺寸，障碍物数量，墙壁颜色，目标点位置等等，这些恰好足够我们设计自主导航功能的训练（感兴趣的可以阅读里面的代码开发更多随机初始化操作，可以使你的模型更好），对于本任务来说只需要修改Content/JsonFiles/EnvGenConfig就可以了。满足数据的domain random，通过每局重新生产成新的随机环境使得model的泛化性能和抗扰性都会增强。

### 4.airsim-rl

这部分是笔者主要写的包，里面将Airsim和UE4的操作api都进行了封装，并形成了gym格式，你可以直接像操作gym环境一样进行step和reset，目前里面设计了DQN，PPO，SAC等算法。

### 5.后续
RL训练自主导航的过程主要是在这个环境中，视频里跟踪移动的行人和避障测试都是在自己搭建的UE4环境里实现的，实现过程也很简单，只是换个工程而已都是通过aisrim的插件实现的。

3D目标检测部分的话是基于https://github.com/lzccccc/SMOKE这个project实现的，安装好smoke后，即可使用3D目标检测+实时导航实现目标追踪效果。

