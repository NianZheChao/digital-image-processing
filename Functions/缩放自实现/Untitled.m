clc   % �������ڵ�����
clear   % ��������ռ�����б���
close all   % �ر����е�Figure����

%д�����ʱ����Ҫ�ڽ����Ͻ���a,b��������

%I = imread('kenan.jpg');   % ��ȡͼƬ
%scale('kenan.jpg', [450,300]);
   % imwrite(uint8(output_img), 'output_img.png'); %���洦����ͼ��clc                                 
I=rgb2gray(imread('kenan.jpg'));
figure,imshow(I); 
Dst=scale(I,3);        %����scale()����
figure,imshow(uint8(Dst));