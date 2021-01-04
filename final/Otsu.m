function T=Otsu(I)
%最大类间方差法
[m,n]=size(I);%得到图像行列像素
I=im2double(I);%变为双精度,即0-1
Th=0;
Thbest=0;
fc=0;
fcmax=0;
count=zeros(256,1); 
pcount=zeros(256,1);
for i=1:m
    for j=1:n
       pixel=I(i,j);
        count(pixel+1)=count(pixel+1)+1;
    end
end
dw=0;
for i=0:255
    pcount(i+1)=count(i+1)/(m*n);
    dw=dw+i*pcount(i+1);
end
while (Th>=0 & Th<=255)
    p1=0;%第一类像素的概率
    ave1=0;%第一类像素的均值
    for i=0:Th
        p1=p1+pcount(i+1);
        ave1=ave1+i*pcount(i+1);
    end
    if p1>0
        ave1=ave1/p1;
    end
    p2=0;%第二类像素的概率
    ave2=0;%第二类像素的均值
    for i=Th+1:255
        p2=p2+pcount(i+1);
        ave2=ave2+i*pcount(i+1);
    end
    if p2>0
        ave2=ave2/p2;
    end
    fc=p1*(ave1-dw)^2+p2*(ave2-dw)^2;%类间方差
    if fc>=fcmax
        fcmax=fc;
        Thbest=Th;
    end
    Th=Th+1;
end
T=Thbest;