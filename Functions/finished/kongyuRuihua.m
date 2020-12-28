%拉普拉斯算子锐化图像，用二阶微分
%四邻接g(x,y)=[f(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)]-4f(x,y)
clear
clc
I1=imread('Einstein.tif');
I=im2double(I1);
[m,n,c]=size(I);
A=zeros(m,n,c);
%分别处理R、G、B
%先对R进行处理
for i=2:m-1
    for j=2:n-1
        A(i,j,1)=I(i+1,j,1)+I(i-1,j,1)+I(i,j+1,1)+I(i,j-1,1)-4*I(i,j,1);
    end
end

%再对G进行处理
for i=2:m-1
    for j=2:n-1
        A(i,j,2)=I(i+1,j,2)+I(i-1,j,2)+I(i,j+1,2)+I(i,j-1,2)-4*I(i,j,2);
    end
end

%最后对B进行处理
for i=2:m-1
    for j=2:n-1
        A(i,j,3)=I(i+1,j,3)+I(i-1,j,3)+I(i,j+1,3)+I(i,j-1,3)-4*I(i,j,3);
    end
end
B=I-A;

 imwrite(B,'lena.tif','tif');
 %imshow('kenan.jpg');title('不清晰图像');figure
 imshow('lena.tif');title('得到的清晰图像')