 clear;
I=imread('apple.jpg');
subplot(121),imshow(uint8(I)),title('原图');
if( ~( size(I,3)-3))%判断是否为彩色图
    I=rgb2gray(I);
end
I=double(I);
[m,n]=size(I);
Smax=0;
for T=0:255   %T为划分区域的阈值            
    sum1=0; num1=0;                   
    sum2=0; num2=0;                   
    for i=1:m
        for j=1:n
            if I(i,j)>=T %区域一
                sum2=sum2+I(i,j); 
                num2=num2+1;              
            else %区域二
                sum1=sum1+I(i,j); 
                num1=num1+1;                
            end 
        end 
    end  
    ave1=sum1/num1; %区域一的灰度均值
    ave2=sum2/num2;%区域二的灰度均值
    S=((ave2-T)*(T-ave1))/(ave2-ave1)^2;%相对距离度量值
    if(S>Smax)%找到最大的距离度量值
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
subplot(122),imshow(I),title('类间最大距离法分割结果');  