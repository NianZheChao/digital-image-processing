clear all
close all
clc
img = imread('kenan.jpg'); %��ȡ����ͼƬ������
B = imresize(img,0.5);
figure, imshow(img), figure, imshow(B)

