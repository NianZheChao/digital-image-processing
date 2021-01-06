%�����䷽�
clear;
I=imread('Einstein.tif');
subplot(211),imshow(uint8(I)),title('ԭͼ');
if( ~( size(I,3)-3))%�ж��Ƿ�Ϊ��ɫͼ
    I=rgb2gray(I);
end
I=double(I);
[m,n]=size(I);
J=Otsu(I);
for i=1:m
    for j=1:n
        if I(i,j)>=J
            I(i,j)=255;
        else
            I(i,j)=0;
       end
    end
end
subplot(2,1,2),imshow(I),title('Otsu��');   

 