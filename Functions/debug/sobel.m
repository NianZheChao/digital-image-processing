%sobel��
I=imread('lenna.jpg'); %��ȡͼ��
 if( ~( size(I,3)-3))%�ж��Ƿ�Ϊ��ɫͼ
    I1 = rgb2gray(I);
 else
     I1=I;
 end
subplot(221),imshow(I),title('ԭͼ');
model=[-1,0,1;
       -2,0,2;
       -1,0,1];
[m,n]=size(I1);
I2=double(I1);
for i=2:m-1
    for j=2:n-1
        I2(i,j)=I1(i+1,j+1)+2*I1(i+1,j)+I1(i+1,j-1)-I1(i-1,j+1)-2*I1(i-1,j)-I1(i-1,j-1);
    end
end
subplot(222),imshow(I2),title('��Ե��ȡ���ͼ��');
I2 = I2 + double(I);
subplot(223),imshow(uint8(I2)),title('�񻯺��ͼ��');
