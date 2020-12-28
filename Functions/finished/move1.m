image = imread('lenna.jpg'); % ��ȡͼ��
[W, H, G] = size(image); % ��ȡͼ���С
image_r=image(:,:,1);
image_g=image(:,:,2);
image_b=image(:,:,3);%��ȡͼ���RGBֵ
res = zeros(W, H, 3); % ����������ÿ�����ص�Ĭ�ϳ�ʼ��Ϊ0����ɫ��
X = 50; % ƽ����X
Y = 50; % ƽ����Y
tras = [1 0 X; 0 1 Y; 0 0 1]; % ƽ�Ƶı任���� 
  for i = 1 : W     
     for j = 1 : H
        temp = [i; j; 1];
        temp = tras * temp; % ����˷�
        x = temp(1, 1);
        y = temp(2, 1);%x��y�ֱ�Ϊͨ������˷��õ����ƽ��λ�õĺ�������ֵ

        % �任���λ���ж��Ƿ�Խ��
        if (x <= W) && (y <= H)&&(x >= 1) && (y >= 1)
            res(x,y,1) = image_r(i, j);
            res(x,y,2) = image_g(i, j);
            res(x,y,3) = image_b(i, j);%���µ�RGBֵ�����ڱ�����   
        end
     end
  end
imshow(uint8(res)); % ��ʾͼ��Ҫ��uint8ת�������¶��ǡ�