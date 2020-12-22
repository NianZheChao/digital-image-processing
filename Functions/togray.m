f=imread('lenna.jpg');
n=double(f);
[x,y,z]=size(f);
R=n(:,:,1);
G=n(:,:,2);
B=n(:,:,3);
g=zeros(x,y);
for i=1:x
    		for j=1:y
       	 g(i,j)=R(i,j)*0.2125+G(i,j)*0.7154+B(i,j)*0.0721;
    		end
end
g=uint8(g);
subplot(1,2,1);imshow(f);title('Ô­Ê¼Í¼Ïñ');
subplot(1,2,2);imshow(g);title('×ª»»Í¼Ïñ');
