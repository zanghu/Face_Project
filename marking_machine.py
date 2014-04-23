#!/usr/bin/env python
#coding=utf-8
from pylearn2.train import Train
from pylearn2.space import Conv2DSpace, VectorSpace
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
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter

from pylearn2.datasets.preprocessing import Pipeline, GlobalContrastNormalization, ZCA #预处理方法

from module.my_mlp_func import AbsoluteTanh #第一个全连接层
from module.my_mlp_func import AbsoluteTanhConvNonlinearity #第二个全连接层，用来作为ConvElemwise的非线性函数

from pylearn2.models.mlp import ConvElemwise #用来替代ConvRecifiedLinear类，特点是可以灵活的更新非线性函数的类型
from pylearn2.models.mlp import Linear #最后一层只有线性变换
from pylearn2.models.mlp import Tanh

from module.yisun import YiSunF1, YiSunEN1, YiSunNM1 #数据集生成类

import numpy
import theano
import theano.tensor as T
import cPickle
import os
import sys
import time

theano.config.floatX = 'float32' #gpu模型

#今后的目标：能够在CPU或GPU机器上单独运行，实现使用python调用c++可执行文件，或者直接使用python调用opencv绘图
#另外希望能够把gpu模型转变成cpu模型
#10000+样本传播一次速度太慢，大约需要60-80+秒，希望能够增加一个模式选项，可以缩减输入样本矩阵的规格，专供debug使用
class MarkingMachine(object):
    """每次调用pylearn2 dataset会消耗较多时间，下一步应该将设计矩阵和关键点矩阵都用.npy文件保存在指定位置，调用时直接读取"""
    
    #初始化中完成的"静态"过程:
    #保存训练集和测试集的类标输出结果
    #保存训练集和测试集的误差矩阵
    
#0.构造函数与工厂函数
    def __init__(self, conv_model, model_name='F1', datatxt=None, axes=('b', 0, 1, 'c')):
        """
        通用型的MarkingMachine，能够实例化为第一层的三个网络模型F1, EN1, NM1
        
        input params:
            conv_model: 模型文件路径，要求是.pkl
            datatxt: 数据路径保存文本
            model_name: 用来指示该模型的类型，同时用于指定保存路径，以便在同时使用多个模型时，可以正确找到文件
            
        object attributes:
            self.model: 对象绑定的conv网络模型
            self.input_space: 对象绑定的输入空间（指明了设计矩阵和样本图像的规格）
            self.view_converter: 和模型输入矩阵配套的view_converter，这样可以在不需要实例化pylearn2 dataset的情况下将设计矩阵转化为topo_view
            self.axes: 为将来可能的样本集预留的接口，一般建议使用默认参数
            self.dict: 数据集和类标目录字典，每个MarkingMachine对象含有6个路径，分别指向6个.npy文件，用于快速调用数据
            self.num_key_points: 模型输出的关键点数目
        """
        assert isinstance(conv_model, MLP)
        self.model = conv_model
        self.model_name = model_name
        
        assert isinstance(self.model.input_space, Conv2DSpace)
        self.input_space = conv_model.input_space
        output_space = conv_model.layers[-1].get_output_space()
        assert isinstance(output_space, VectorSpace)
        print 'output_space.dim:', output_space.dim
        self.num_key_points = output_space.dim / 2
        
        self.shape = (self.input_space.shape[0], self.input_space.shape[1], self.input_space.num_channels)
        print 'self.shape:', self.shape
        self.axes = axes;
        view_converter = DefaultViewConverter(self.shape, axes=self.axes) #不创建YiSunF1对象，只创建一个相同功能的view_converter
        self.view_converter = view_converter
        
        d = {}
        f = open(datatxt)
        data_path_list = [line.strip('\n') for line in f.readlines()]
        assert len(data_path_list) == 7
        #gcn和zca过得数据集
        d['train_input_path'] = data_path_list[0] #'/home/zanghu/yisun_cvpr2013/static_data/F1_data/F1_X_gcn_zca_train.npy'
        d['test_input_path'] = data_path_list[1] #'/home/zanghu/yisun_cvpr2013/static_data/F1_data/F1_X_gcn_zca_test.npy'
        #原始数据集，暂时没什么用
        d['train_orin_path'] = data_path_list[2] #'/home/zanghu/yisun_cvpr2013/static_data/F1_data/F1_X_orin_train.npy'
        d['test_orin_path'] = data_path_list[3] #'/home/zanghu/yisun_cvpr2013/static_data/F1_data/F1_X_orin_test.npy'
    
        d['train_label_path'] = data_path_list[4] #'/home/zanghu/yisun_cvpr2013/static_data/F1_data/F1_y_float_train.npy'
        d['test_label_path'] = data_path_list[5] #'/home/zanghu/yisun_cvpr2013/static_data/F1_data/F1_y_float_test.npy'
        
        d['save_dir'] = os.path.join(data_path_list[6], self.model_name)
        if (not os.path.exists(d['save_dir'])) or (not os.path.isdir(d['save_dir'])):
            os.makedirs(d['save_dir'])
        
        self.dict = d
        
        #计算并保存测试集和训练集输出
        print 'calculating outputs...'; sys.stdout.flush()
        self.get_train_test_output_label(savetxt=True, savenpy=True)
        #self.get_tarin_test_ground_truth()
        #计算训练集和测试集距离误差
        print 'calculating train/test error...'; sys.stdout.flush()
        self.cal_train_test_error()
        #计算并保存'成功-失败'矩阵
        print 'calculating failure rate...'; sys.stdout.flush()
        self.cal_train_test_failure(threshold=None, percent=0.05, savetxt=True)
        #计算平均失败率
        print 'failure rate is: ',  self.get_train_test_failure_rate(mode='average', threshold=None, percent=0.05)
        
    @classmethod
    def get_machine(cls, model_path, data_path=None):
        """工厂函数，用来使用模型路径产生实例"""
        if ((not os.path.exists(model_path)) or (not os.path.isfile(model_path))):
            print "model path error"
            return
        f = open(model_path)
        model = cPickle.load(f)
        f.close()
        return MarkingMachine(conv_model=model, datatxt=data_path)
    
#1.通用的基本操作方法========================================================
    def get_train_orin(self):
        """原子操作"""
        return numpy.load(self.dict['train_orin_path'])
    
    def get_test_orin(self):
        """原子操作"""
        return numpy.load(self.dict['test_orin_path'])
    
    def get_train_label_float(self):
        """原子操作"""
        return numpy.load(self.dict['train_label_path'])
    
    def get_test_label_float(self):
        """原子操作"""
        return numpy.load(self.dict['test_label_path'])
    
    def get_train_input(self):
        """原子操作"""
        return numpy.load(self.dict['train_input_path'])
    
    def get_test_input(self):
        """原子操作"""
        return numpy.load(self.dict['test_input_path'])
        
    def __get_func(self):
        """获得网络正向传播的theano函数"""
        x = T.tensor4(dtype='float32') #今后可以尝试去掉float32限制，只用theano.config.floatX=float32来控制
        f = theano.function([x], self.model.fprop(x))
        return f
        
        
    def apply(self, image_design_matrix, is_gpu=True):
        """使用模型获得关键点"""
        t0 = time.clock()
        assert isinstance(image_design_matrix, numpy.ndarray)
        if image_design_matrix.ndim == 1: #将单个样本向量转化为1xn的样本矩阵
            image_design_matrix = image_design_matrix.reshape(1, -1)
        assert image_design_matrix.shape[1] == numpy.prod(self.shape) #检验样本尺寸，只支持与self.shape的尺寸相同的输入
        if is_gpu and image_design_matrix.dtype != 'float32': #gpu模型，类型转换
            image_design_matrix = numpy.cast['float32'](image_design_matrix)
        image_topo_view = self.view_converter.design_mat_to_topo_view(image_design_matrix)
        assert image_topo_view.ndim == 4
        func = self.__get_func()
        label_matrix_output = func(image_topo_view)
        assert label_matrix_output.shape == (image_design_matrix.shape[0], self.num_key_points*2)
        print 'time elapsed on fprop:', time.clock() - t0 ; sys.stdout.flush()
        
        return label_matrix_output
    
    def get_output_label(self, design_matrix, savetxt=False, save_dir=None):
        """通用的获得关键点矩阵的方法"""
        assert isinstance(design_matrix, numpy.ndarray)
        assert design_matrix.ndim == 2
        assert design_matrix.shape[1] == self.input_space.shape[0] * self.input_space.shape[1] * self.input_space.num_channels
        label_train = self.apply(design_matrix)
        if savetxt:
            if save_dir is None:
                numpy.savetxt('./label.txt', label_train, fmt='%.12f')
            else:
                numpy.savetxt((os.path.join(save_dir, 'label.txt')), label_train, fmt='%.12f')
        return label
    
    @staticmethod #该方法没有用到任何对象参数或者类参数，因此定义为staticmethod
    def cal_error(Y, Y_hat):
        """计算每个输出点坐标与对应的真实关键点坐标的距离，输出一个矩阵"""
        delta_square = numpy.square(Y - Y_hat)
        dist_square = delta_square[::2] + delta_square[1::2]
        dist = numpy.sqrt(dist_square)
        return dist
    
#2.针对第一类数据的静态方法=================================
    
    def show(self, image_path=None, design_matrix_path=None):
        """希望实现输入一张图片或者图片文件夹的位置，显示一幅加入关键点的图片"""
        return None
    
    def get_train_test_output_label(self, savetxt=False, savenpy=False):
        """获得以训练集/测试集为输入时，模型输出的关键点矩阵"""
        img_matrix_train = numpy.load(self.dict['train_input_path']) #dsy_train.X #gpu模型
        img_matrix_test = numpy.load(self.dict['test_input_path'])
        label_train = self.apply(img_matrix_train)
        label_test = self.apply(img_matrix_test)
        self.label_train_output = label_train #添加对象属性，将输出结果保存到当前对象
        self.label_test_output = label_test #添加对象属性，将输出结果保存到当前对象
        if savetxt:
            numpy.savetxt((os.path.join(self.dict['save_dir'], 'train_label_output.txt')), label_train, fmt='%.12f')
            print 'train_label saved'
            numpy.savetxt((os.path.join(self.dict['save_dir'], 'test_label_output.txt')), label_test, fmt='%.12f')
            print 'test_label saved'
        if savenpy:
            numpy.save((os.path.join(self.dict['save_dir'], 'train_label_output.npy')), label_train)
            print 'train_label saved'
            numpy.save((os.path.join(self.dict['save_dir'], 'test_label_output.npy')), label_test)
            print 'test_label saved'
        return [label_train, label_test]
    
    def get_tarin_test_ground_truth(self):
        """返回真实的类标矩阵"""
        train_ground_truth = self.get_train_label_float()
        test_ground_truth = self.get_test_label_float()
        #self.label_train_true = train_ground_truth #结果挂接到模型
        #self.label_test_true = test_ground_truth
        return [train_ground_truth, test_ground_truth]
    
    def cal_train_test_error(self):
        """计算训练集/测试集的误差（与真实点的距离）矩阵"""
        if not hasattr(self, 'label_train_output') or not hasattr(self, 'label_test_output'):
            self.get_train_test_output_label()
        train_label_true = self.get_train_label_float()
        test_label_true = self.get_test_label_float()
        #计算误差距离矩阵
        error_train = self.cal_error(train_label_true, self.label_train_output)
        error_test = self.cal_error(test_label_true, self.label_test_output)
        #加入对象属性
        self.error_train = error_train
        self.error_test = error_test
        
    def cal_train_test_failure(self, threshold=None, percent=0.05, prefix='', savetxt=False):
        """由距离误差矩阵以及判定阈值，获得每个元素取0-1值的'成功-失败'矩阵，考虑到评估准则的变化，决定不保存计算结果"""
        if threshold is None:
            #l is the width of the bounding box returned by our face detector
            l = self.input_space.shape[1]
            threshold = l * percent
        if (not hasattr(self, 'error_train')) or (not hasattr(self, 'error_test')):
            self.cal_train_test_error()
        #failure matrix中值为1的点为失败点，值为1的点未成功点
        failure_matrix_train = (self.error_train >= threshold)
        failure_matrix_test = (self.error_test >= threshold)
        if savetxt:
            numpy.savetxt(os.path.join(self.dict['save_dir'], prefix+'failure_matrix_train.txt'), failure_matrix_train, fmt='%i')
            numpy.savetxt(os.path.join(self.dict['save_dir'], prefix+'failure_matrix_test.txt'), failure_matrix_test, fmt='%i')
        
        return [failure_matrix_train, failure_matrix_test]
    
    def get_train_test_failure_rate(self, mode='average', threshold=None, percent=0.05):
        """"""
        failure_matrix_train, failure_matrix_test = self.cal_train_test_failure(threshold=threshold, percent=percent)
        
        train_denominator = numpy.prod(failure_matrix_train.shape, dtype=theano.config.floatX)
        train_nominator = numpy.sum(failure_matrix_train, dtype=theano.config.floatX)
        train_failure_rate = train_nominator / train_denominator
        
        test_denominator = numpy.prod(failure_matrix_test.shape, dtype=theano.config.floatX)
        test_nominator = numpy.sum(failure_matrix_test, dtype=theano.config.floatX)
        test_failure_rate = test_nominator / test_denominator
        
        return [train_failure_rate, test_failure_rate]
        
        
#3.动态方法，用于满足用户请求=======================================
    
    def cal_failure(self, Y, Y_hat, threshold=None, percent=0.05, savetxt=False, prefix='', save_dir='./'):
        """
        由距离误差矩阵以及判定阈值，获得每个元素取0-1值的'成功-失败'矩阵
        prefix: 用来区分文件名，避免出现重名文件相互覆盖的情况
        """
        if threshold is None:
            #l is the width of the bounding box returned by our face detector
            l = self.input_space.shape[1]
            threshold = l * percent
        dist_matrix = MarkingMachine.cal_error(Y, Y_hat)
        #failure matrix中值为1的点为失败点，值为1的点未成功点
        failure_matrix = (dist_matrix >= threshold)
        
        if savetxt:
            numpy.savetxt(os.path.join(save_dir, prefix+'failure_matrix.txt'), failure_matrix, fmt='%i')
        
        return failure_matrix
    
    def get_failure_rate(self, Y, Y_hat, mode='average', threshold=None, percent=0.05):
        """
        统计失败图片数的模式有多种, 目前只实现'average'模式
        mode: 'average'，即分母总数为全体关键点个数'样本数x关键点数'
        """
        failure_matrix = self.cal_failure(Y, Y_hat, threshold=threshold, percent=percent);
        denominator = numpy.prod(failure_matrix.shape, dtype=theano.config.floatX)
        nominator = numpy.sum(failure_matrix, dtype=theano.config.floatX)
        failure_rate = nominator / denominator
        
        return failure_rate     
    
    #def cal_key_point_error(self, name='LE'):
    #    """一旦模型训练完成，那么每次输入一组样本，将模型输出与样本真实的关键点坐标比较，一次性保存结果供之后查询"""
    #    train_output_label_matrix = self.get_output_label_matrix('train')
    #    test_output_label_matrix = self.get_output_label_matrix('test')
        
    def show_img(self, which_set='train', start=0, stop=12):
        """测试self.view_converter是否正确配置"""
        dsy = YiSunF1(which_set=which_set, gcn_zca=False)
        topo_view = self.view_converter.design_mat_to_topo_view(dsy.X[start:stop, :])
        rows = numpy.cast[int](numpy.ceil(numpy.sqrt(stop - start)))
        cols = 3
        topo_view = numpy.cast['uint8'](topo_view)
        print 'topo_view.shape:', topo_view.shape
        img = numpy.zeros((39, 39, 3), dtype='uint8')
        for i in xrange(4):
            for j in xrange(3):
                #print 'topo_view[i*cols+j].shape:', topo_view[i*cols+j].shape
                
                #pylab.imshow(img)
                #pylab.show()
                #print 'max:', numpy.max(topo_view[i*cols+j])
                #print 'max:', numpy.min(topo_view[i*cols+j])
                #f = open('/home/zanghu/topo_view.txt', 'w')
                #for a in xrange(39):
                #    for b in xrange(39):
                #        for k in xrange(3):
                            #print topo_view[i*cols+j][a, b, k]
                #            img[a, b, k] = topo_view[i*cols+j][a, b, k]
                #pylab.imshow(img)
                #pylab.show()
                #print 'rows:', rows
                #print 'cols;', cols
                print i*3+j
                pylab.subplot(3, 4, i*3+j+1); pylab.axis('off'); pylab.imshow(topo_view[i*cols+j])
        pylab.show()
        
if __name__ == '__main__':
    import numpy
    marking_machine = MarkingMachine.get_machine(model_path='/home/zanghu/Pro_Datasets/yisun/train/cpu_model/convolutional_network_best_F1.pkl', \
                                                 data_path='/home/zanghu/Pro_Datasets/yisun/train/static_data/data.txt')
    #test_label_matrix = marking_machine.get_test_label(savetxt=True)
    #train_label_matrix = marking_machine.get_train_label(savetxt=True)
    #marking_machine.show_img()
    print marking_machine.shape
    
    #marking_machine.get_failure_rate(marking_machine.get_
    
