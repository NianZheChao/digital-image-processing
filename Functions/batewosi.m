close all;
clear all;
f = imread('pic.tif');
f = mat2gray(f,[0 255]);

[M,N] = size(f);
P = 2*M;
Q = 2*N;
fc = zeros(M,N);

for x = 1:1:M
    for y = 1:1:N
        fc(x,y) = f(x,y) * (-1)^(x+y);
    end
end

F = fft2(fc,P,Q);

H_0 = zeros(P,Q);
H_1 = zeros(P,Q);
H_2 = zeros(P,Q);

for x = (-P/2):1:(P/2)-1
     for y = (-Q/2):1:(Q/2)-1
        D = (x^2 + y^2)^(0.5);
        %n=1
        D_0 = 10;
        H_0(x+(P/2)+1,y+(Q/2)+1) = 1/(1+(D/D_0)^2);
        D_0 = 50;
        H_1(x+(P/2)+1,y+(Q/2)+1) = 1/(1+(D/D_0)^2);   
        D_0 = 150;
        H_2(x+(P/2)+1,y+(Q/2)+1) = 1/(1+(D/D_0)^2);
        %H_2(x+(P/2)+1,y+(Q/2)+1) = 1/(1+(D/D_0)^6);
     end
end

G_0 = H_0 .* F;
G_1 = H_1 .* F;
G_2 = H_2 .* F;

g_0 = real(ifft2(G_0));
g_0 = g_0(1:1:M,1:1:N);

g_1 = real(ifft2(G_1));
g_1 = g_1(1:1:M,1:1:N);

g_2 = real(ifft2(G_2));
g_2 = g_2(1:1:M,1:1:N);         

for x = 1:1:M
    for y = 1:1:N
        g_0(x,y) = g_0(x,y) * (-1)^(x+y);
        g_1(x,y) = g_1(x,y) * (-1)^(x+y);
        g_2(x,y) = g_2(x,y) * (-1)^(x+y);
    end
end

%% -----show-------
figure();
subplot(1,4,1);
imshow(f,[0 1]);
xlabel('1).原图');

subplot(1,4,2);
imshow(g_0,[0 1]);
xlabel('5).效果图(D=10)');

subplot(1,4,3);
imshow(g_1,[0 1]);
xlabel('8).效果图(D=50)');


subplot(1,4,4);
imshow(g_2,[0 1]);
xlabel('11).效果图(D=150)');