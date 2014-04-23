#!/usr/bin/env python
#coding=utf-8
"""
"""
from pylearn2.train import Train
from pylearn2.space import Conv2DSpace
from pylearn2.datasets.mnist import MNIST
from pylearn2.models.mlp import MLP, Softmax #, ConvRectifiedLinear
from pylearn2.models.mlp import LinearOutput
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.costs.cost import SumOfCosts, MethodCost
from pylearn2.costs.mlp import WeightDecay
from pylearn2.termination_criteria import MonitorBased
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.training_algorithms.sgd import MomentumAdjustor
from pylearn2.termination_criteria import EpochCounter

from pylearn2.datasets.preprocessing import Pipeline, GlobalContrastNormalization, ZCA #预处理方法

from module.my_mlp_func import AbsoluteTanh #第一个全连接层
from module.my_mlp_func import AbsoluteTanhConvNonlinearity #第二个全连接层，用来作为ConvElemwise的非线性函数

from pylearn2.models.mlp import ConvElemwise #用来替代ConvRecifiedLinear类，特点是可以灵活的更新非线性函数的类型
from pylearn2.models.mlp import Linear #最后一层只有线性变换
from pylearn2.models.mlp import Tanh

from module.yisun import YiSunF1, YiSunEN1, YiSunNM1 #数据集生成类

#1.读取数据集
dsy_train = YiSunF1(which_set='train', start=0, stop=10000, gcn_zca=True, y_float=True, one_zero=False)
dsy_test = YiSunF1(which_set='test', start=0, stop=3466, gcn_zca=True, y_float=True, one_zero=False)

#2.数据集预处理
#pipeline = Pipeline()
#pipeline.items.append(GlobalContrastNormalization(sqrt_bias=10., use_std=True))
#pipeline.items.append(ZCA())
#dsy_train.apply_preprocessor(preprocessor=pipeline, can_fit=True)
#dsy_test.apply_preprocessor(preprocessor=pipeline, can_fit=True)

#3.初始化mlp的每一层
#三个标准卷积层：卷积 + 非线性 + max-pooling
#注意到由于非线性部分的非线性函数一般都是单调的（虽然不一定是严格单调的），所以非线性和pooling的先后顺序对于本层网络的输出是无影响的
crl_layer_h1 = ConvElemwise(layer_name='h1_crl', kernel_shape=[4, 4], kernel_stride=[1, 1], output_channels=20, irange=0.05, 
                                pool_shape=[2, 2], pool_stride=[2, 2], pool_type='max', 
                                nonlinearity=AbsoluteTanhConvNonlinearity(), max_kernel_norm=1.9365)
crl_layer_h2 = ConvElemwise(layer_name='h2_crl', kernel_shape=[3, 3], kernel_stride=[1, 1], output_channels=40, irange=0.05, 
                                pool_shape=[2, 2], pool_stride=[2, 2], pool_type='max', 
                                nonlinearity=AbsoluteTanhConvNonlinearity(), max_kernel_norm=1.9365)
crl_layer_h3 = ConvElemwise(layer_name='h3_crl', kernel_shape=[3, 3], kernel_stride=[1, 1], output_channels=60, irange=0.05, 
                                pool_shape=[2, 2], pool_stride=[2, 2], pool_type='max',
                                nonlinearity=AbsoluteTanhConvNonlinearity(),max_kernel_norm=1.9365)

#一个非标准卷积层，只有卷积和非线性变换，没有pooling，实现方式是直接将pooling层的shape和stride都设为(1, 1)
crl_layer_h4 = ConvElemwise(layer_name='h4_crl', kernel_shape=[2, 2], kernel_stride=[1, 1], output_channels=80, irange=0.05, 
                                pool_shape=[1, 1], pool_stride=[1, 1], pool_type='max', 
                                nonlinearity=AbsoluteTanhConvNonlinearity(), max_kernel_norm=1.9365)

#一个带有非线性的全连接层
fc_layer_h5 = Tanh(dim=120, layer_name='h5_fc', irange=0.05)

#一个只有线性变化的输出层 
fc_layer_h6 = LinearOutput(dim=10, layer_name='h6_fc', irange=0.05); #当使用该层作为最后一层时，mlp应该可以使用get_default_cost()

#4.制造mlp
layers = [crl_layer_h1, crl_layer_h2, crl_layer_h3, crl_layer_h4, fc_layer_h5, fc_layer_h6]
print layers[-1].layer_name

mlp_model = MLP(batch_size=100, input_space=Conv2DSpace(shape=[39, 39], num_channels=3, axes=('b', 0, 1, 'c')), layers=layers)

method_cost = MethodCost(method='cost_from_X') #可以保留
#调用顺序：mlp.cost_from_X -> mlp.cost -> mlp.laters[-1].cost(Y, Y_hat) -> linear.cost_from_cost_matrix()

weight_decay = WeightDecay(coeffs=[0.005, 0.005, 0.005, 0.005, 0.005, 0.005]) #每一层都使用相同的惩罚系数是否合理?
cost = SumOfCosts([method_cost, weight_decay])

monitoring_dataset = {'train': dsy_train, 'test': dsy_test}


alg = SGD(batch_size=100, learning_rate=0.0001, init_momentum=0.5, monitoring_dataset=monitoring_dataset, cost=cost,\
          termination_criterion=EpochCounter(max_epochs=1000))

extensions = [MonitorBasedSaveBest(channel_name='test_objective', save_path='concolutional_network_best_F1.pkl'), \
                                   MomentumAdjustor(start=1, saturate=10, final_momentum=0.9)]

train = Train(dataset=dsy_train, model=mlp_model, algorithm=alg, extensions=extensions)

train.main_loop()
