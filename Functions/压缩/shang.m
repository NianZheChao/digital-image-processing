function h = shang(x, n)
error(nargchk(1, 2, nargin)); % Check input arguments
if nargin < 2
    n = 256;  % Default for n
end

x = double(x); % Make input double
xh = hist(x(:), n); % Compute N-bin histogram
xh = xh / sum(xh(:)); % Compute probabilities

% Make mask to eliminate 0's since log2(0) = -inf.
i = find(xh); %µÈ¼ÛÓÚfind(xh ~= 0)
h = -sum(xh(i) .* log2(xh(i)));  % Compute entropy
