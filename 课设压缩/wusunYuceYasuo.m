%冈萨雷斯，数字图像处理（MATLAB）（第二版）教材例6.5
clear;
clc;
f = imread('cat.tif');
e = mat2lpc(f);  %对f进行预测编码处理
%-------------------------------------%
fprintf('熵的比较');
e_entropy = ntrop(e) %计算e的熵
f_entropy = ntrop(f) %计算f的熵

%-------------------------------------%
fprintf('压缩比率的比较');
%直接对原始图像进行霍夫曼压缩
hf = mat2huff(f);
hr = imratio(f, hf)
%对预测编码图像进行霍夫曼压缩
c = mat2huff(e);
cr = imratio(f, c)

%显示效果
%预测误差e的直方图
[h, x] = hist(e(:)*512, 512);
figure;
subplot(1,2,1); imshow(mat2gray(e)); title('预测误差图像');
subplot(1,2,2); bar(x,h,'k'); title('预测误差直方图');
