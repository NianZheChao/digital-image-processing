function y = quantize(x, b, type)
%QUANTIZE Quantizes the elements of a UINT8 matrix.
% Y = QUANTIZE(X, B, TYPE) quantizes X to B bits. Truncation is used unless
% TYPE is 'igs' for Improved Gray Scale quantization

narginchk(2, 3);
if ~ismatrix(x) || ~isreal(x) || ~isnumeric(x) || ~isa(x, 'uint8')
    error('The input must be a UINT8 numeric matrix.');
end

% Create bit masks for the quantization
lo = uint8(2 ^ (8-b) - 1); %lo有8-b位1在低位，其余位为0
hi = uint8(2 ^ 8 - double(lo) - 1); %hi有b位1在高位，其余位为0

% Perform standard quantization unless IGS is specified
if nargin < 3 || ~strcmpi(type, 'igs') %进行一般灰度量化
    y = bitand(x, hi); %计算x和hi的按位与运算，即保留像素灰度的高位值，灰度差异的大小体现在高位
    
% Else IGS quantization. Process column-wise. If the MSB's of the pixel are
% all 1's, the sum is set to the pixel value. Else, add the pixel value to
% the LSB's of the previous sum. Then take the MSB's of the sum as the
% quantized value.
% MSB：最高有效位；LSB：最低有效位
else %进行IGS量化
    %IGS扰动方法：对于任何高位不是hi的像素，将其加上前一列相邻像素的对应lo低位的值
    [m,n] = size(x);
    s = zeros(m, 1);
    %获得对应于x中高位不是hi的那些元素的集合
    hitest = double(bitand(x, hi) ~=  hi);
    %figure;imshow(hitest);
    x  = double(x);
    for j = 1:n
        tt = double(bitand(uint8(s), lo)); 
        %tt为经IGS扰动后的X中前一列像素的对应于lo低位的值
        s = x(:,j)+hitest(:,j) .* tt; %经扰动后的当前列的像素值
        y(:, j) = bitand(uint8(s), hi);%保留当前列像素灰度的高位值
    end
end