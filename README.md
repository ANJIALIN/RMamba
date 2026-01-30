# (To protect privacy, we will publish our RMamba algorithm and the link to the dataset we used once the article is accepted.)

Disturbances Robust Mamba Model for Industrial Anomaly Diagnosis

### Abstract

Conventional lightweight deep learning approaches often struggle to capture dynamic operational characteristics under resource-limited computational conditions. Recently, Mamba has emerged as a promising lightweight architecture and demonstrated notable success across multiple domains, owing to its inherent capability in modeling dynamic dependencies. However, its state-space mechanism exhibits high sensitivity to intense disturbances, undermining model robustness for industrial anomaly diagnosis. To address this issue, a novel robust Mamba (RMamba) framework is proposed for reliable industrial anomaly diagnosis under strong disturbance conditions. Specificallly, an
attention Mamba module is designed to jointly extract dynamic and static features. Then a lightweight global feature extraction module is built to re-calibrate global representations via channelwise adaptive scaling and shifting, enabling holistic feature perception with negligible computational overhead. Meanwhile, a dynamic gating module with adaptive learning that performs convolutional sparsification on global features is constructed to simultaneously enhance computational efficiency and improve robustness under lightweight design constraints. Extensive experiments conducted on a public conveyor belt video dataset and an industrial smelting furnace validate the efficacy of the proposed method.

## Table of Contents
- [Environment](#Environment)
- [Dataset](#Dataset)

## Environment

ps:This work uses the most basic Python configuration and does not require too many complicated operations. The mentioned frameworks are all in AttMamba.py. Audiences can extract them by themselves if necessary. Of course, the most basic pip framework needs to be installed by yourself. <img src="make images/2.png" width="40"/>

注：本作品使用环境都是最基础的python配置，不需要过多复杂的操作，所提框架都在AttMamba.py中，各位观众老爷如有需要可自行提取。当然，最基本的pip框架各位观众老爷要自己安装哦. <img src="make images/2.png" width="40"/>

### If you have a Mamba environment, look here
If your computer can run the Mamba environment directly, right-click to open the terminal and execute the following command:
```
jupyter lab
```
Then run test-code.ipynb directly. <img src="make images/1.png" width="40"/>

### Of course, if you don’t have a Mamba environment on your computer, don’t be afraid. Let me help you solve this problem.

First, you need to configure the Docker environment: 🐳 

```bash
https://www.docker.com/
```

Then, to ensure that you can download correctly, you need to pull the image of mamba:

```bash
git clone https://github.com/your_username/vision-mamba-docker.git
cd vision-mamba-docker
```

Next, bulid Docker iamge:

```bash
docker build -t vision-mamba .
```

Finally, run the docker container:

```bash
docker run --gpus all -it --name mamba-container -v $(pwd):/workspace vision-mamba
```

Then you can do the same as above.

## Dataset
ps: The link to the public dataset mentioned is as follows. This is someone else's. If you want to use it in a paper, you must cite it, or you will be punished：<img src="make images/ad.jpg" width="40"/>


注：所提到的公开数据集链接如下，这个是人家的，如果要使用发论文要引用，要不小心挨打。<img src="make images/ad.jpg" width="40"/>

```bash
https://ljf1113.github.io/IPAD_VAD/
```
I just used the conveyor belt dataset, if you want to verify my data, you need to pay attention. Of course, I have verified it for you in jupyter lab.<img src="make images/xg.jpg" width="40"/>


## Citation
My paper has not been published yet, so I'll leave you alone for now.<img src="make images/hx.jpg" width="40"/>

```bibtex
If my work is helpful to you, please remember to cite it when my paper is published.

如果我的东西对你有帮助，等我论文发表了，记得引用哦。
```
