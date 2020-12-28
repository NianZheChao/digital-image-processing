function [image_out] = GaoSiDiTongLvBoQi(image_in,D0)
%GLPFΪ��˹��ͨ�˲�����D0Ϊ��ֹƵ��
%����Ϊ��Ҫ���и�˹��ͨ�˲��ĻҶ�ͼ�����Ϊ�����˲�֮��ĻҶ�ͼ��
f=image_in;
f=im2double(f);
% 1������һ����СΪM��N������ͼ��f(x,y)���õ�������P=2M��Q=2N
M=size(f,1);   N=size(f,2);
P=2*M;          Q=2*N; 
% 2��������ͼ��f(x,y)��ӱ�Ҫ������0,�γɴ�СΪP��Q�����ͼ��fp(x,y)
fp=zeros(P,Q);
fp(1:M,1:N)=f(1:M,1:N);
% 3���ã�-1��^(x+y)����fp(x,y)�Ƶ���任������
for i=1:P
    for j=1:Q
        fp(i,j)=(-1)^(i+j)*double(fp(i,j));
    end
end
% 4���������Բ���3��ͼ���DFT���õ�F��u,v��
F=fft2(fp,P,Q);
% 5������һ��ʵ�ġ��ԳƵ��˲�����H(u,v)�����СΪP��Q�������ڣ�P/2��Q/2����������������γɳ˻�G��u,v��=H(u,v)F(u,v)
H=zeros(P,Q);
a=2*(D0^2);
for u=1:P
    for v=1:Q
        D=(u-P/2)^2+(v-Q/2)^2;
        H(u,v)=exp(-D./a);
    end
end
G=F.*H; %Ƶ�����˲�
% 6���õ�������ͼ��
gp=ifft2(G); %Ƶ��ת����ʱ��ͼ��
gp=real(gp);
for i=1:P
    for j=1:Q
        gp(i,j)=(-1)^(i+j)*double(gp(i,j));
    end
end
% 7��ͨ����gp(x,y)������������ȡM��N���򣬵õ����մ�����g(x,y)
image_out=gp(1:M,1:N);  
end
