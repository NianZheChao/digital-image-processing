f=imread('pic.tif');
q=quantize(f,4,'igs');%�ú���quantize����igs������4bit
qs=double(q)/16;
e=mat2lpc(qs);    %ʹ��Ԥ������
c=mat2huff(e);     %��ʹ�û���������
imratio(f,c)
subplot(131),imshow(f),title('ԭʼͼ��');
subplot(132),imshow(e),title('ʹ��Ԥ������');

%ne=huff2mat(c);   %���л���������
%nqs=lpc2mat(ne);  %ͼ���һά����Ԥ�����
%nq=16*nqs;
%subplot(131),imshow(ne),title('����������');
%subplot(132),imshow(nqs),title('����Ԥ�����');
%subplot(133),imshow(nq),title('16��������Ԥ�����')
%compare(q,nq)
%rmse=compare(f,nq)