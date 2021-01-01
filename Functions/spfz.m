clear ;
img = imread('apple.jpg');
[H,W,Z] = size(img); % ��ȡͼ���С
I=im2double(img);%��ͼ������ת����˫����
res = ones(H,W,Z); % ����������ÿ�����ص�Ĭ�ϳ�ʼ��Ϊ1����ɫ��
tras = [1 0 0; 0 -1 W; 0 0 1]; % ˮƽ����ı任���� 
for x0 = 1 : H
    for y0 = 1 : W
        temp = [x0; y0; 1];%��ÿһ���λ�ý��л���
        temp = tras * temp; % �����㷨���У�����˷���ת���������ԭ����λ��
        x1 = temp(1, 1);%�µ�����x1λ��
        y1 = temp(2, 1);%�µ�����y1λ��
        % �任���λ���ж��Ƿ�Խ��
        if (x1 <= H) & (y1 <= W) & (x1 >= 1) & (y1 >= 1)%�µ���λ��ҪС���µ���λ��
            res(x1,y1,:)= I(x0,y0,:);%����ͼ����ɫ��ֵ
        end
    end
end
subplot(1,2,1), imshow(I);
subplot(1,2,2), imshow(res);