f=imread('pic.tif');
q=quantize(f,4,'igs');%用函数quantize进行igs量化到4bit
qs=double(q)/16;
e=mat2lpc(qs);    %使用预测编码后
c=mat2huff(e);     %再使用霍夫曼编码
imratio(f,c)
subplot(131),imshow(f),title('原始图像');
subplot(132),imshow(e),title('使用预测编码后');

%ne=huff2mat(c);   %进行霍夫曼解码
%nqs=lpc2mat(ne);  %图像的一维线性预测解码
%nq=16*nqs;
%subplot(131),imshow(ne),title('霍夫曼解码');
%subplot(132),imshow(nqs),title('线性预测解码');
%subplot(133),imshow(nq),title('16倍的线性预测解码')
%compare(q,nq)
%rmse=compare(f,nq)