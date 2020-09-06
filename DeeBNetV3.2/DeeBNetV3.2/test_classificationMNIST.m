 %Test classification in DBN in MNIST data set
clc;
% clear all;
res={};
more off;
addpath(genpath('DeepLearnToolboxGPU'));
addpath('DeeBNet');
data = MNIST.prepareMNIST('D:\MatlabFiles\DeeBNetV3.2\DataSets\Image\MNIST\');%using MNIST dataset completely.
% data = MNIST.prepareMNIST_Small('+MNIST\');%uncomment this line to use a small part of MNIST dataset.
data.normalize('minmax');
data.shuffle();
data.validationData=data.testData;  %���Լ�
data.validationLabels=data.testLabels;  %���Լ���Ӧ�ı�ǩ
dbn=DBN('classifier');  %ʹ��DBN���з���
% RBM1
rbmParams=RbmParameters(500,ValueType.binary);
rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.PCD;
rbmParams.performanceMethod='reconstruction';
rbmParams.maxEpoch=50;
dbn.addRBM(rbmParams);
% RBM2
rbmParams=RbmParameters(500,ValueType.binary);
rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.PCD;
rbmParams.performanceMethod='reconstruction';
rbmParams.maxEpoch=50;
dbn.addRBM(rbmParams);
% RBM3
rbmParams=RbmParameters(2000,ValueType.binary);
rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.PCD;
rbmParams.maxEpoch=50;
rbmParams.rbmType=RbmType.discriminative;
rbmParams.performanceMethod='classification';
dbn.addRBM(rbmParams);
%train
ticID=tic;
dbn.train(data);
toc(ticID)    %��ʱ��������ʱ�����Ԥѵ������ʱ��
%test train
classNumber=dbn.getOutput(data.testData,'bySampling');
errorBeforeBP=sum(classNumber~=data.testLabels)/length(classNumber)
%BP
ticID=tic;
dbn.backpropagation(data);  %���򴫲�������DBN���еķ���
toc(ticID);
%test after BP
classNumber=dbn.getOutput(data.testData);   %��ò������ݵ����
errorAfterBP=sum(classNumber~=data.testLabels)/length(classNumber)  %����������ݵ����
