 I=imread('apple.jpg'); 
if( ~( size(I,3)-3))%�ж��Ƿ�Ϊ��ɫͼ
    I1 = rgb2gray(I);
 else
     I1=I;
 end
subplot(221);
imshow(I1),title('ԭͼ');
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
subplot(222),imshow(uint8(I2)),title('��Ե��ȡ���ͼ��');
I2 = double(I2)+ double(I1);
subplot(223),imshow(uint8(I2)),title('�񻯺��ͼ��');
