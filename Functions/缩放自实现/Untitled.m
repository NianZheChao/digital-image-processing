clc   % 清除命令窗口的内容
clear   % 清除工作空间的所有变量
close all   % 关闭所有的Figure窗口

%写界面的时候，需要在界面上接受a,b两个参数

%I = imread('kenan.jpg');   % 读取图片
%scale('kenan.jpg', [450,300]);
   % imwrite(uint8(output_img), 'output_img.png'); %保存处理后的图像clc                                 
I=rgb2gray(imread('kenan.jpg'));
figure,imshow(I); 
Dst=scale(I,3);        %调用scale()函数
figure,imshow(uint8(Dst));