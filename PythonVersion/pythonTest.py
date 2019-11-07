import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
# torch.cuda.device_count()
# torch.cuda.get_device_name(0)

#计算cupy矩阵乘法以及其计算平均时间
#shape:矩阵维度，shape*shape
#times:计算次数，计算时间是取平均时间
def cupy_test(shape,times):
    sumT = 0
    for i in range(0,times):
        arr_gpu=cp.random.rand(shape,shape,dtype=cp.float)
        start = time.clock()
        cp.einsum('ij, jk',arr_gpu,arr_gpu)
        end = time.clock()
        if i == 0:
            continue
        sumT += (end-start)
        #print((end-start)*1000)
    times-=1
    avgT = ((sumT/times)*1000)
    print("cupy avg time %f"%avgT)
    return avgT


#计算torch_gpu矩阵乘法以及其计算平均时间
#size:size*size
#times:计算次数，计算时间是取平均时间
def torch_gpu_test(size,times):
    sumT = 0
    for i in range(0,times):
        # 随机生成矩阵，然后通过cuda拷贝到gpu上
        a = torch.rand(size,size).cuda()
        b = torch.rand(size,size).cuda()
        start = time.clock()
        c = torch.mm(a, b)
        end = time.clock()
        if i == 0:
            continue
        #print((end-start)*1000)
        sumT += end-start
    times-=1
    avgT = ((sumT/times)*1000)
    print("torch gpu avg time %f"%avgT)
    return avgT


#计算numpy矩阵乘法以及其计算平均时间
#shape:shape*shape
#times:计算次数，计算时间是取平均时间
def numpy_test(shape,times):
    sumT = 0
    for i in range(0,times):
        arr_cpu=np.random.rand(shape,shape)
        start = time.clock()
        np.einsum('ij, jk',arr_cpu,arr_cpu)
        end = time.clock()
        if i == 0:
            continue
        sumT += (end-start)
        #print((end-start)*1000)
    times-=1
    avgT = ((sumT/times)*1000)
    print("cupy avg time %f"%avgT)
    return avgT


# 根据上述计算的时间画图
def draw_time(timeDict):
    sizeList = timeDict['size']
    plt.plot(sizeList,timeDict['cupy'],marker='o',label='cupy')
    plt.plot(sizeList,timeDict['pytorch-gpu'],marker='o',label='pytorch-gpu')
    #plt.plot(sizeList,timeDict['numpy'],marker='o',label='numpy')
    plt.xlabel('matrix size') 
    plt.ylabel('cal time (ms)')
    plt.legend()
    plt.show()


# 测试矩阵大小从2^0 到2^12各种方法的计算时间
def statistics_time():
    times = 2
    timeDict = {}
    timeDict['cupy'] = []
    timeDict['pytorch-gpu'] = []
    timeDict['numpy'] = [] 
    timeDict['size'] = []
    for n in range(0,13):
        size = 2**n
        print(size)
        cT = cupy_test(size,times)
        tT = torch_gpu_test(size,times)
        nT = numpy_test(size,times)
        timeDict['cupy'] .append(cT)
        timeDict['pytorch-gpu'].append(tT)
        timeDict['numpy'].append(nT)
        timeDict['size'].append(size)
    return timeDict

if  __name__ == "__main__":
    #设置gpu
    torch.cuda.set_device(0)  
    timeDict = statistics_time()
    draw_time(timeDict)    