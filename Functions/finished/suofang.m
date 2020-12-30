clear all
close all
clc
img = imread('kenan.jpg'); %读取输入图片的数据
B = imresize(img,0.5);
figure, imshow(img), figure, imshow(B)

