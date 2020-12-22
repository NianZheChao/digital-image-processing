img = imread('b.png');
%R通道
R = img(:,:,1);
%G通道
G = img(:,:,2);
%B通道
B = img(:,:,3);
%Alpha通道
[I,map,Alpha] = imread('b.png');
background = imread('backgroundB.png');
%计算参数
a = Alpha/255;
%三通道合成
img2(:,:,1) = img(:,:,1) .* a + (1-a) .* background(:,:,1);
img2(:,:,2) = img(:,:,2) .* a + (1-a) .* background(:,:,2);
img2(:,:,3) = img(:,:,3) .* a + (1-a) .* background(:,:,3);
imshow(img2);
imwrite(img2,'combineB.png')
