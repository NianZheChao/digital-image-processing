%������˹������ͼ����MATLAB�����ڶ��棩�̲���6.5
clear;
clc;
f = imread('cat.tif');
e = mat2lpc(f);  %��f����Ԥ����봦��
%-------------------------------------%
fprintf('�صıȽ�');
e_entropy = ntrop(e) %����e����
f_entropy = ntrop(f) %����f����

%-------------------------------------%
fprintf('ѹ�����ʵıȽ�');
%ֱ�Ӷ�ԭʼͼ����л�����ѹ��
hf = mat2huff(f);
hr = imratio(f, hf)
%��Ԥ�����ͼ����л�����ѹ��
c = mat2huff(e);
cr = imratio(f, c)

%��ʾЧ��
%Ԥ�����e��ֱ��ͼ
[h, x] = hist(e(:)*512, 512);
figure;
subplot(1,2,1); imshow(mat2gray(e)); title('Ԥ�����ͼ��');
subplot(1,2,2); bar(x,h,'k'); title('Ԥ�����ֱ��ͼ');
