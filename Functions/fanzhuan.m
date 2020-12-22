f = imread('test.jpg');

[R, C] = size(f);
res = zeros(R, C);

for i = 1 : R
    for j = 1 : C
        x = i;
        y = C - j + 1;
        res(x, y) = f(i, j);
    end
end

imshow(uint8(res));
