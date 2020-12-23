img = imread('Einstein.tif');
background = imread('test.jpg');
img = double(img);
background = double(background);
%Rͨ��
R = img(:,:,1);
%Gͨ��
G = img(:,:,2);
%Bͨ��
B = img(:,:,3);
%Alphaͨ��
[I,map,Alpha] = imread('Einstein.tif');

%�������
a = Alpha/255;
%��ͨ���ϳ�
img2(:,:,1) = img(:,:,1) .* a + (1-a) .* background(:,:,1);
img2(:,:,2) = img(:,:,2) .* a + (1-a) .* background(:,:,2);
img2(:,:,3) = img(:,:,3) .* a + (1-a) .* background(:,:,3);
imshow(img2);
imwrite(img2,'combineB.png')
