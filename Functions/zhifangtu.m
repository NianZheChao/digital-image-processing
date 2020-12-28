I=imread('test1.jpg');     %读入图片
J=rgb2gray(I);                 %将彩色图片转换为灰度图
graydis=zeros(1,256);           %设置矩阵大小
graydispro=zeros(1,256);
new_graydis=zeros(1,256);
new_graydispro=zeros(1,256);
[h w]=size(J);
new_tu=zeros(h,w);
%计算原始直方图各灰度级像素个数graydis
for x=1:h
     for y=1:w
        graydis(1,I(x,y))=graydis(1,I(x,y))+1;
     end
end
%计算原始直方图graydispro
graydispro=graydis./sum(graydis);
subplot(2,2,1);
plot(graydispro);
title('灰度直方图');
xlabel('灰度值');ylabel('像素的概率密度');
%计算原始累计直方图
for i=2:256
    graydispro(1,i)=graydispro(1,i)+graydispro(1,i-1);
end
%计算和原始灰度对应的新的灰度t[]，建立映射关系
for i=1:256
t(1,i)=floor(254*graydispro(1,i)+0.5);
end
%统计新直方图各灰度级像素个数new_graydis
for i=1:256
    new_graydis(1,t(1,i)+1)=new_graydis(1,t(1,i)+1)+graydis(1,i);
end
%计算新的灰度直方图new_graydispro
new_graydispro=new_graydis./sum(new_graydis);
subplot(2,2,2);
plot(new_graydispro);
title('均衡化后的灰度直方图');
xlabel('灰度值');ylabel('像素的概率密度');
%计算直方图均衡后的新图new_tu
for x=1:h
    for y=1:w
      new_tu(x,y)=t(1,I(x,y));
    end
end
subplot(2,2,3);
imshow(J);
title('原图');
subplot(2,2,4);
imshow(new_tu,[]);
title('直方图均衡化后的图');
