 clear;
I=imread('apple.jpg');
subplot(121),imshow(uint8(I)),title('ԭͼ');
if( ~( size(I,3)-3))%�ж��Ƿ�Ϊ��ɫͼ
    I=rgb2gray(I);
end
I=double(I);
[m,n]=size(I);
Smax=0;
for T=0:255   %TΪ�����������ֵ            
    sum1=0; num1=0;                   
    sum2=0; num2=0;                   
    for i=1:m
        for j=1:n
            if I(i,j)>=T %����һ
                sum2=sum2+I(i,j); 
                num2=num2+1;              
            else %�����
                sum1=sum1+I(i,j); 
                num1=num1+1;                
            end 
        end 
    end  
    ave1=sum1/num1; %����һ�ĻҶȾ�ֵ
    ave2=sum2/num2;%������ĻҶȾ�ֵ
    S=((ave2-T)*(T-ave1))/(ave2-ave1)^2;%��Ծ������ֵ
    if(S>Smax)%�ҵ����ľ������ֵ
        Smax=S;
        Th=T;
     end
end
for i=1:m
    for j=1:n
        if I(i,j)>=Th
            I(i,j)=255;
        else
            I(i,j)=0;
        end
    end
end
subplot(122),imshow(I),title('��������뷨�ָ���');  