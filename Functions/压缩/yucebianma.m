%Ô¤²â±àÂë
function y = yucebianma(x, f)
error(nargchk(1, 2, nargin));   % Check input arguments
if nargin < 2                   % Set default filter if omitted
    f = 1;
end

x = double(x);                  % Ensure double for computations
[m, n] = size(x);               % Get dimensions of input matrix
p = zeros(m, n);                % Init linear prediction to 0
xs = x; zc = zeros(m, 1);       % Prepare for input shift and pad

for j = 1 : length(f)           % For each filter coefficient
    xs =[zc xs(:, 1:end - 1)];  % Shift and zero pad x
    p = p + f(j) * xs;          % Form partial prediction sums
end

y = x - round(p);               % Compute prediction error