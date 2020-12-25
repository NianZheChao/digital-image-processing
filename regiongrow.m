function J=regiongrow(I)
if isinteger(I)
    I=im2double(I);%uint8转double
end
imshow(I)
[M,N]=size(I);
[y,x]=getpts;%单击取点后，按enter结束
x1=round(x);%取整
y1=round(y);
seed=I(x1,y1);
J=zeros(M,N);
J(x1,y1)=1;
sum=seed;
suit=1;%点的个数
count=1;
threshold=0.15;
while count>0
    s=0;
    count=0;
    for i=1:M%遍历图像
        for j=1:N
            if J(i,j)==1
                if (i-1)>0&&(i+1)<(M+1)&&(j-1)>0&&(j+1)<(N+1)%判断此点是否为图像边界上的点，是边界点就算了，没有8邻域（很粗糙，有5邻域啊）
                    for u=-1:1%遍历8邻域
                        for v=-1:1
                            if J(i+u,j+v)==0&&abs(I(i+u,j+v)-seed)<=threshold%判断是否需要标记，1、未被标记过，2、和标记区灰度值之差小于阈值
                                J(i+u,j+v)=1;
                                count=count+1;
                                s=s+I(i+u,j+v);
                            end
                        end
                    end
                end
            end
        end
    end
    suit=suit+count;
    sum=sum+s;
    seed=sum/suit;%取所有已标记区的平均值作为下一次的计算基准
end
end

