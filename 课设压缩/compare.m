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
    %histΪƵ��ֱ��ͼ������
    %emaxΪ������ͳ�Ʒ�Χ����������[1, emax]��Χ(����1��ԭ��)�ȷ�Ϊemax������,
    %ͳ������e��Ԫ���䵽��Щ���������
    %����ֵx��hΪһά���顣xΪ�����ϰ�����˳�����е��Ա�����hΪ��Ӧ�Ա����������ϵ�Ӧ����
    [h,x] = hist(e(:), emax);
    if length(h) >= 1
        figure; bar(x, h, 'k');
        % Scale the error image symmetrically and display
        emax = emax / scale;
        %��double�������eת����ȡֵ��ΧΪ[-emax emax]�ĻҶ�ͼ��
        %���У�����-emax��ֵת��Ϊ0������emax��ֵת��Ϊ1.
        e = mat2gray(e, [-emax emax]);
        figure; imshow(e);
    end
end