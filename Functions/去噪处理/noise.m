I=imread('lenna.jpg');
J=imnoise(I,'salt & pepper'); %��������
K=imnoise(I,'gaussian',0,0.05);%�����ֵΪ0������Ϊ0.005�ĸ�˹���� 
after=midfilt(J,3);  %��ֵ�˲�
after1=avg_filter(J,3);  %��ֵ�˲�
after2=midfilt(K,3); 
after3=avg_filter(K,3); 
subplot(3,3,1);
imshow(I);
title('ԭͼ��');
subplot(3,3,2);
imshow(J);
title('���뽷��������');
subplot(3,3,3); imshow(K);
title('�����˹����֮���ͼ��');
subplot(3,3,4);
imshow(after);
title('��ֵ�˲����ͼ��1');
subplot(3,3,5);
imshow(after1);
title('��ֵ�˲����ͼ��1');
subplot(3,3,6);
imshow(after2);
title('��ֵ�˲����ͼ��2');
subplot(3,3,7);
imshow(after3);
title('��ֵ�˲����ͼ��2');
