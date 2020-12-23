function g=rotate(f,angle)
[h,w,d]=size(f);
radian=angle/180*pi;
cosa=cos(radian);
sina=sin(radian);

w2=round(abs(cosa)*w+h*abs(sina));
h2=round(abs(cosa)*h+w*abs(sina));
g=uint8(zeros(h2,w2,3));
for x=1:w2
    for y=1:h2
        x0=uint32(x*cosa+y*sina-0.5*w2*cosa-0.5*h2*sina+0.5*w);
        y0=uint32(y*cosa-x*sina+0.5*w2*sina-0.5*h2*cosa+0.5*h);
        
        x0=round(x0);
        y0=round(y0);
        if x0>0 && y0>0 && w>=x0 && h>=y0
            g(y,x,:)=f(y0,x0,:);
        end
    end
end
