img = imread('apple.jpg');
[H,W,Z] = size(img); % 获取图像大小
I=im2double(img);%将图像类型转换成双精度
res = ones(H,W,Z); % 构造结果矩阵。每个像素点默认初始化为1（白色）
tras = [-1 0 H; 0 1 0; 0 0 1]; % 垂直镜像的变换矩阵
for x0 = 1 : H
    for y0 = 1 : W
        temp = [x0; y0; 1];%将每一点的位置进行缓存
        temp = tras * temp; % 根据算法进行，矩阵乘法：转换矩阵乘以原像素位置
        x1 = temp(1, 1);%新的像素x1位置
        y1 = temp(2, 1);%新的像素y1位置
        % 变换后的位置判断是否越界
        if (x1 <= H) & (y1 <= W) & (x1 >= 1) & (y1 >= 1)%新的行位置要小于新的列位置
            res(x1,y1,:)= I(x0,y0,:);%进行颜色赋值
        end
    end
end
subplot(1,2,1), imshow(I) ;
subplot(1,2,2), imshow(res);