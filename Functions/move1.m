image = imread('lenna.jpg'); % 读取图像
[W, H, G] = size(image); % 获取图像大小
image_r=image(:,:,1);
image_g=image(:,:,2);
image_b=image(:,:,3);%获取图像的RGB值
res = zeros(W, H, 3); % 构造结果矩阵。每个像素点默认初始化为0（黑色）
X = 50; % 平移量X
Y = 50; % 平移量Y
tras = [1 0 X; 0 1 Y; 0 0 1]; % 平移的变换矩阵 
  for i = 1 : W     
     for j = 1 : H
        temp = [i; j; 1];
        temp = tras * temp; % 矩阵乘法
        x = temp(1, 1);
        y = temp(2, 1);%x、y分别为通过矩阵乘法得到后的平移位置的横纵坐标值

        % 变换后的位置判断是否越界
        if (x <= W) && (y <= H)&&(x >= 1) && (y >= 1)
            res(x,y,1) = image_r(i, j);
            res(x,y,2) = image_g(i, j);
            res(x,y,3) = image_b(i, j);%将新的RGB值赋予在背景上   
        end
     end
  end
imshow(uint8(res)); % 显示图像，要用uint8转化，以下都是。