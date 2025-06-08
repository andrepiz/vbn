function rr_origin2observer = nonlineartriangulateration(loses, ranges, rr_origin2beacons, x0)
% Non-Linear Triangulateration: Estimate observer position
% by solving nonlinear system:
%   p_i - x = r_i * d_i
% using nonlinear least squares.
%
% Inputs:
%   loses - 3xN matrix of unit LOS vectors d_i
%   ranges      - 1xN vector of distances r_i
%   rr_origin2beacons     - 3xN matrix of beacon rr_origin2observeritions p_i
%   x0          - 3x1 initial guess (optional, default mean estimate)
%
% Output:
%   rr_origin2observer - 3x1 estimated observer position

if nargin < 4
    % Use mean estimator as initial guess
    x0 = triangulateration(loses, ranges, rr_origin2beacons);
end

options = optimoptions('lsqnonlin','Display','off','TolFun',1e-8);

% Residual function
fun = @(x) residuals(x, ranges, rr_origin2beacons, loses);

% Solve nonlinear least squares
rr_origin2observer = lsqnonlin(fun, x0, [], [], options);

end

function r = residuals(x, ranges, rr_origin2beacons, loses)
% Residuals for nonlinear least squares
% r_i = p_i - x - r_i * d_i (vectorized over all rr_origin2beacons)

diff = rr_origin2beacons - x; % 3xN
r = diff - loses .* ranges; % 3xN

r = r(:); % convert to vector for lsqnonlin
end
