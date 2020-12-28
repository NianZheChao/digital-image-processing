clear;
clc;
f = imread('cat.tif');
c = mat2huff(f);
cr1 = imratio(f, c)

save SqueezeCat c;
cr2 = imratio('cat.tif', 'SqueezeCat.mat')