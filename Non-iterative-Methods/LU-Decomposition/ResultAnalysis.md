## LU分解

### 实验内容

$$
\left\{\begin{array}{l}\boldsymbol{L} \boldsymbol{y}=\boldsymbol{b} \\ \boldsymbol{U} \boldsymbol{x}=\boldsymbol{y}\end{array}\right.
$$

1. 使用C实现right-looking算法以及LU分解后的$\boldsymbol{y},\boldsymbol{x}$的求解的CPU版本
2. 使用CUDA实现right-looking算法以及LU分解后的$\boldsymbol{y},\boldsymbol{x}$的求解的并行GPU版本（加速思路在之前的报告中有详细说明，具体实现方式没有用到新的技巧，所以就不在报告中阐述了）
3. 在不同维度以及单双精度下所生成的随机可逆矩阵，对GPU和CPU版本算法进行对比测试
4. 记录测试数据，分析测试结果

### 实验结果与分析

#### 实验现象和分析

实验现象：

+ 在实验过程中，在小维度下所得到的GPU版本和CPU版本计算结果间的误差范围基本小于0.01，基本保证了GPU算法实现的正确性。但在不断提高维度的过程中，由于GPU版本在每次迭代中都会有误差，而维度越大则迭代次数越多，迭代所造成的累积误差越来越严重，最终导致计算结果与CPU版本有严重误差（甚至达到inf）。

+ 在单精度下GPU版本中一开始迭代的那些列中里CPU和GPU的结果基本保持在6位数精度范围内相等，因为单精度float的精度差不多是6-7位。

+ 在单精度下GPU的加速比要大于双精度，同时其计算结果于CPU相比误差也更大，因为双精度double的精度范围16位左右。

造成精度误差的可能原因：

1. 生成的随机可逆矩阵不能进行LU分解
2. 在LU分解前没有进行行初等变换（即$PLU$）

3. 迭代所造成的累积误差

#### 实验结果与分析

实验数据都记录在time.xlsx中，计算时间对比图还没画

结果分析：

1. LU分解计算复杂度$\mathrm{O}\left(\mathrm{n}^{\wedge} 3\right)$，在GPU版的每轮迭代中，一个block负责subMatrix中一行的更新，而一个block包含了1024个thread，所以并行性已经很高了，而且算法本身的计算复杂度是很高的，所以能够获得较高的加速比。
2. $\boldsymbol{y},\boldsymbol{x}$的求解复杂度$\mathrm{O}\left(\mathrm{n}^{\wedge} 2\right)$，在求解过程中数据间关联较高且有大量的规约操作，导致GPU实现的版本加速效果不明显。

###  实验环境

####  硬件环境

- **CPU:** Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz（12核）
- **GPU:** NVIDIA Tesla K40c (11G)

#### 软件环境

- Nvidia Cuda 8.0.61

#### 运行环境

- 命令行 通过nvcc编译代码

### 实验改进

1. 使用matlab或者C/C++的库生成可一定能LU分解的可逆矩阵，看看精度误差是否依旧严重
2. 算法上解决精度误差问题：超出能力范围，且没有思路