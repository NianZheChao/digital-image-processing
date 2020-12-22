I=imread('lenna.jpg');
J=imnoise(I,'salt & pepper'); %椒盐噪声
K=imnoise(I,'gaussian',0,0.05);%加入均值为0，方差为0.005的高斯噪声 
after=midfilt(J,3);  %中值滤波
after1=avg_filter(J,3);  %均值滤波
after2=midfilt(K,3); 
after3=avg_filter(K,3); 
subplot(3,3,1);
imshow(I);
title('原图像');
subplot(3,3,2);
imshow(J);
title('加入椒盐噪声后');
subplot(3,3,3); imshow(K);
title('加入高斯噪声之后的图像');
subplot(3,3,4);
imshow(after);
title('中值滤波后的图像1');
subplot(3,3,5);
imshow(after1);
title('均值滤波后的图像1');
subplot(3,3,6);
imshow(after2);
title('中值滤波后的图像2');
subplot(3,3,7);
imshow(after3);
title('均值滤波后的图像2');
