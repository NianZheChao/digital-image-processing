function rmse = compare(f1, f2, scale)
%COMPARE Computes and displays the error between two matrices.
% RMSE = COMPARE(F1, F2, SCALE) returns the root-mean-square error between
% inputs F1 and F2, displays a histogram of the difference, and displays a
% scaled difference image. When SCALE is omitted, a scale factor of 1 is
% used.

% Check input arguments and set defaults
error(nargchk(2,3,nargin));
if nargin < 3
    scale = 1;
end

% Compute the root-mean-square error.
e = double(f1) - double(f2);
[m,n] = size(e);
rmse = sqrt(sum(e(:).^2) / (m * n));

% Output error image & histogram if an error(i.e., rmse -= 0).
if rmse
    % Form error histogram.
    emax = max(abs(e(:)));
    %hist为频数直方图函数。
    %emax为纵轴上统计范围，即把纵轴[1, emax]范围(坐标1在原点)等分为emax个区间,
    %统计数组e中元素落到这些区间的数量
    %返回值x和h为一维数组。x为横轴上按递增顺序排列的自变量，h为对应自变量的纵轴上的应变量
    [h,x] = hist(e(:), emax);
    if length(h) >= 1
        figure; bar(x, h, 'k');
        % Scale the error image symmetrically and display
        emax = emax / scale;
        %把double类的数组e转换成取值范围为[-emax emax]的灰度图像
        %其中，等于-emax的值转换为0，等于emax的值转换为1.
        e = mat2gray(e, [-emax emax]);
        figure; imshow(e);
    end
end