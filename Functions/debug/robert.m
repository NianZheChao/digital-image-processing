 I=imread('apple.jpg'); %读取图像
 if( ~( size(I,3)-3))%判断是否为彩色图
    I1 = rgb2gray(I);
 else
     I1=I;
 end
subplot(221),imshow(I1),title('原图');
model=[0,-1;1,0];
[m,n]=size(I1);
I2=double(I1);
for i=2:m-1
    for j=2:n-1
        I2(i,j)=I1(i+1,j)-I1(i,j+1);
    end
end
subplot(222),imshow(I2),title('边缘提取后的图像');
I2 = I2 + double(I1);
subplot(223),imshow(uint8(I2)),title('锐化后的图像');
