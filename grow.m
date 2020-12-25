I=imread('grow.png');
figure;subplot(121);imshow(I);title('原始图像');
I=double(I);              %转换为灰度值是0-1的双精度
[M,N]=size(I);            %得到原图像的行列数
[y,x]=getpts;             %获得区域生长起始点
x1=round(x);              %横坐标取整
y1=round(y);              %纵坐标取整
seed=I(x1,y1);            %将生长起始点灰度值存入seed中
Y=zeros(M,N);             %作一个全零与原图像等大的图像矩阵Y，作为输出图像矩阵
Y(x1,y1)=1;               %将Y中与所取点相对应位置的点设置为白点
sum=seed;                 %储存符合区域生长条件的点的灰度值的总和
suit=1;                   %储存符合区域生长条件的点的总个数
count=1;                  %每次判断一点周围八点符合条件的新点的数目
threshold=10;             %域值，即某一点与周围八点的绝对差值要小于阈值
while count>0             %判断是否有新的符合生长条件的点，若没有，则结束
s=0;                      %判断一点周围八点时，符合条件的新点的灰度值之和
count=0;
for i=1:M
   for j=1:N
     if Y(i,j)==1
      if (i-1)>0 && (i+1)<(M+1) && (j-1)>0 && (j+1)<(N+1) %判断此点是否为图像边界上的点
       for u= -1:1                                        %判断点周围八点是否符合域值条件
        for v= -1:1                                       %u,v为偏移量
          if  Y(i+u,j+v)==0 && abs(I(i+u,j+v)-seed)<=threshold%判断是否未存在于输出矩阵Y，并且为符合域值条件的点
             Y(i+u,j+v)=1;                                %符合以上两条件即将其在Y中与之位置对应的点设置为白点
             count=count+1;                               %新的、符合生长条件的点的总个数
             s=s+I(i+u,j+v);                              %新的、符合生长条件的点的总灰度数
          end
        end  
       end
      end
     end
   end
end
suit=suit+count;                                   %目前区域所有符合生长条件的点的总个数
sum=sum+s;                                         %目前区域所有符合生长条件的点的总灰度值
seed=sum/suit;                                     %计算新的灰度平均值
end

subplot(122);imshow(Y);title('分割后图像');