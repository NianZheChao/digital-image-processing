I=imread('test1.jpg');     %����ͼƬ
J=rgb2gray(I);                 %����ɫͼƬת��Ϊ�Ҷ�ͼ
graydis=zeros(1,256);           %���þ����С
graydispro=zeros(1,256);
new_graydis=zeros(1,256);
new_graydispro=zeros(1,256);
[h w]=size(J);
new_tu=zeros(h,w);
%����ԭʼֱ��ͼ���Ҷȼ����ظ���graydis
for x=1:h
     for y=1:w
        graydis(1,I(x,y))=graydis(1,I(x,y))+1;
     end
end
%����ԭʼֱ��ͼgraydispro
graydispro=graydis./sum(graydis);
subplot(2,2,1);
plot(graydispro);
title('�Ҷ�ֱ��ͼ');
xlabel('�Ҷ�ֵ');ylabel('���صĸ����ܶ�');
%����ԭʼ�ۼ�ֱ��ͼ
for i=2:256
    graydispro(1,i)=graydispro(1,i)+graydispro(1,i-1);
end
%�����ԭʼ�Ҷȶ�Ӧ���µĻҶ�t[]������ӳ���ϵ
for i=1:256
t(1,i)=floor(254*graydispro(1,i)+0.5);
end
%ͳ����ֱ��ͼ���Ҷȼ����ظ���new_graydis
for i=1:256
    new_graydis(1,t(1,i)+1)=new_graydis(1,t(1,i)+1)+graydis(1,i);
end
%�����µĻҶ�ֱ��ͼnew_graydispro
new_graydispro=new_graydis./sum(new_graydis);
subplot(2,2,2);
plot(new_graydispro);
title('���⻯��ĻҶ�ֱ��ͼ');
xlabel('�Ҷ�ֵ');ylabel('���صĸ����ܶ�');
%����ֱ��ͼ��������ͼnew_tu
for x=1:h
    for y=1:w
      new_tu(x,y)=t(1,I(x,y));
    end
end
subplot(2,2,3);
imshow(J);
title('ԭͼ');
subplot(2,2,4);
imshow(new_tu,[]);
title('ֱ��ͼ���⻯���ͼ');
