clc
clear all
close all
warning off all;
% load Features
load f1
load f2
load f3
load f4
load f5
load f6
load f7
load f8
load f9
load f10
load f11
load f12
A=[Fea1 Fea2 Fea3 Fea4 Fea5 Fea6 Fea7 Fea8 Fea9 Fea10 Fea11 Fea12]./1000;
%specify the input and the targets
x=[1 2 3 2 3 3 2 2 4 4 4 4];

%create a feed forward neural network
net1 = newff(minmax(A),[60,6,1],{'tansig','tansig','purelin'},'trainrp');
net1.trainParam.show = 1000;
net1.trainParam.lr = 0.04;
net1.trainParam.epochs = 700;
net1.trainParam.goal = 1e-5;

%train the neural network using the input,target and the created network
[netan] = train(net1,A,x);
%save the network
save netan

%simulate the network for a particular input
y = round(abs((sim(netan,A))))