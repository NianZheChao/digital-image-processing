function cr = imratio(f1, f2)
%IMRATIO Computes the ratio of the bytes in two images/variables
% CR = IMRATIO(F1, F2) returns the ratio of the number of bytes in
% variables/files F1 and F2. If F1 and F2 are an original and compressed
% image, respectively, CR is the compression ratio.
error(nargchk(2, 2, nargin));  %check input argument
cr = bytes(f1) / bytes(f2);    %Compute the ratio

%.......................................................................
function b = bytes(f)
%Return the number of bytes in input f. If f is a string. assume that it is
%an image filename; if not, it is an image vraiable.
if ischar(f)
    info = dir(f);
    b = info.bytes;
elseif isstruct(f)
    %MATLAB is whos function reports an extra 124 bytes of memory per
    %structure field because of the way MATLAB stores structures in memory.
    %Don't count this extra memory; instead, add up the memory associated
    %with each field.
    b = 0;
    fields = fieldnames(f);  %fildnames函数获得结构体f中各字段的名称
    k1 = length(fields);
    for k = 1:k1
        elements = f.(fields{k});
        k2 = length(elements);
        for m = 1:k2
            ele = elements(m);
            b = b + bytes(ele);
        end
    end
else
    info = whos('f'); %whos函数返回携带matlab工作空间中变量'f'信息的结构体
    b = info.bytes;
end