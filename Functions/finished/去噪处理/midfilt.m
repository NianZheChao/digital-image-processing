function d=midfilt(x, n)
[M,N]=size(x);
x1=x;
x2=x1;
for i=1:M-n+1
    for j=1:N-n+1
        c=x1(i:i+n-1,j:j+n-1);
        e=c(1,:); 
        for k=2:n
            e=[e,c(k,:)];
        end
        x2(i+(n-1)/2,j+(n-1)/2)=median(e);

    end
end
d=x2;
