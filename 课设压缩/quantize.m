function y = quantize(x, b, type)
%QUANTIZE Quantizes the elements of a UINT8 matrix.
% Y = QUANTIZE(X, B, TYPE) quantizes X to B bits. Truncation is used unless
% TYPE is 'igs' for Improved Gray Scale quantization

narginchk(2, 3);
if ~ismatrix(x) || ~isreal(x) || ~isnumeric(x) || ~isa(x, 'uint8')
    error('The input must be a UINT8 numeric matrix.');
end

% Create bit masks for the quantization
lo = uint8(2 ^ (8-b) - 1); %lo��8-bλ1�ڵ�λ������λΪ0
hi = uint8(2 ^ 8 - double(lo) - 1); %hi��bλ1�ڸ�λ������λΪ0

% Perform standard quantization unless IGS is specified
if nargin < 3 || ~strcmpi(type, 'igs') %����һ��Ҷ�����
    y = bitand(x, hi); %����x��hi�İ�λ�����㣬���������ػҶȵĸ�λֵ���ҶȲ���Ĵ�С�����ڸ�λ
    
% Else IGS quantization. Process column-wise. If the MSB's of the pixel are
% all 1's, the sum is set to the pixel value. Else, add the pixel value to
% the LSB's of the previous sum. Then take the MSB's of the sum as the
% quantized value.
% MSB�������Чλ��LSB�������Чλ
else %����IGS����
    %IGS�Ŷ������������κθ�λ����hi�����أ��������ǰһ���������صĶ�Ӧlo��λ��ֵ
    [m,n] = size(x);
    s = zeros(m, 1);
    %��ö�Ӧ��x�и�λ����hi����ЩԪ�صļ���
    hitest = double(bitand(x, hi) ~=  hi);
    %figure;imshow(hitest);
    x  = double(x);
    for j = 1:n
        tt = double(bitand(uint8(s), lo)); 
        %ttΪ��IGS�Ŷ����X��ǰһ�����صĶ�Ӧ��lo��λ��ֵ
        s = x(:,j)+hitest(:,j) .* tt; %���Ŷ���ĵ�ǰ�е�����ֵ
        y(:, j) = bitand(uint8(s), hi);%������ǰ�����ػҶȵĸ�λֵ
    end
end