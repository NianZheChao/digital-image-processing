clear;
image = imread('Einstein.tif');
[width,height]=size(image);

result2 = image;
k1=0.05;
k2=0.05;
a1=rand(width,height)<k1;
a2=rand(width,height)<k2;
t1=result2(:,:,1);
t1(a1&a2)=0;
t1(a1& ~a2)=255;
result2(:,:,1)=t1;
imshow(result2);