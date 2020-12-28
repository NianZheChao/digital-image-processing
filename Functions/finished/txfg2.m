%Ostu
I=imread('lenna.jpg');
I=rgb2gray(I);
I=double(I);
subplot(2,1,1),imshow(uint8(I)),title('Ô­Í¼');
[m,n]=size(I);
Th=Otsu(I);
for i=1:m
    for j=1:n
        if I(i,j)>=Th
            I(i,j)=255;
        else
            I(i,j)=0;
       end
    end
end
subplot(2,1,2),imshow(I),title('Otsu·¨');   

 