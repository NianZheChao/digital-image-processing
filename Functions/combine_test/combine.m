img1=imread('2.jpg');
img2=imread('1.jpg');


[des1, des2] = siftMatch(img1, img2);
des1=[des1(:,2),des1(:,1)];%左右（x和y）交换 % 为过滤匹配准备参数
des2=[des2(:,2),des2(:,1)];%

%用 基础矩阵F 过滤匹配的特征点对
matchs = matchFSelect(des1, des2) %匹配位置索引（掩码）
des1=des1(matchs,:);%取出内点
des2=des2(matchs,:);

% 画出匹配特征点的连接线（好点）
drawLinedCorner(img1,des1,img2, des2) ;


    [H,W,k]=size(img1);%图像大小
    l_r=W-des1(1,2)+des2(1,2);%只取水平方向（第一个匹配点）重叠宽度
     
     
    % 1、直接拼接-------------------------------------------------
     
    %[H,W,k]=size(img1);
    %l_r=405;%重叠宽度（W-宽 至 W）---如果不用特征匹配这里直接写重合区宽
    L=W+1-l_r;%左边起点
    R=W;%右边尾点
    n=R-L+1;%重叠宽度：就是l_r
    %直接拼接图
    im=[img1,img2(:,n:W,:)];%1全图+2的后面部分
    figure;imshow(im);title('直接拼接图');
    
    