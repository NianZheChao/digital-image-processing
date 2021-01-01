 I=imread('apple.jpg'); 
if( ~( size(I,3)-3))%判断是否为彩色图
    I1 = rgb2gray(I);
 else
     I1=I;
 end
subplot(221);
imshow(I1),title('原图');
model=[-1,0,1;
       -1,0,1;
       -1,0,1];
[m,n]=size(I1);
I2=I1;
for i=2:m-1
    for j=2:n-1
        tem=I1(i-1:i+1,j-1:j+1);
        tem=double(tem).*double(model);
        I2(i,j)=sum(sum(tem));   
         end
end
subplot(222),imshow(uint8(I2)),title('边缘提取后的图像');
I2 = double(I2)+ double(I1);
subplot(223),imshow(uint8(I2)),title('锐化后的图像');
