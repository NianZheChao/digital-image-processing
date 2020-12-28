I=imread('test.jpg');
[M,N]=size(I);
I_shuiping=I;
for i=1:M
    for j=1:N
        I_shuiping(i,j)=I(i,N-j+1);
    end
end
subplot(2,2,1);
imshow(I);
subplot(2,2,2);
imshow(I_shuiping);

I_chuizhi=I;
for i=1:M
    for j=1:N
        I_chuizhi(i,j)=I(M-i+1,j);
    end
end
subplot(2,2,3);
imshow(I_chuizhi);


