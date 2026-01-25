function rr_origin2observer = nonlineartriangulateration(loses, ranges, rr_origin2beacons, x0, flag_use_weighted_residuals, w_los, w_range)
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
if flag_use_weighted_residuals
    fun = @(x) weighted_residuals(x, ranges, rr_origin2beacons, loses, w_los, w_range);
else
    fun = @(x) residuals(x, ranges, rr_origin2beacons, loses);
end

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

function r = weighted_residuals(x, ranges, rr_origin2beacons, loses, w_los, w_range)
% Residuals weighted to reflect different uncertainties

diff = rr_origin2beacons - x; % 3xN, from estimated pos to each beacon
r_vec = diff - loses .* ranges; % vector residuals: p_i - x - r_i * d_i

% Weight each component
% LOS direction residuals: perpendicular to d_i
% Range residuals: along d_i

N = size(rr_origin2beacons, 2);
r = zeros(3*N, 1);

for i = 1:N
    d = loses(:, i);
    err_vec = r_vec(:, i);

    % Project residual onto d (range component) and perpendicular (LOS component)
    range_error = dot(err_vec, d);
    los_error = err_vec - range_error * d;

    % Pick arbitrary vector not parallel to d
    if abs(d(1)) < abs(d(2))
        temp = [1;0;0];
    else
        temp = [0;1;0];
    end
    
    u1 = temp - dot(temp, d) * d;  % make orthogonal to d
    u1 = u1 / norm(u1);            % normalize
    
    u2 = cross(d, u1);             % ensure orthogonal and right-handed
    u2 = u2 / norm(u2);

    los_proj = [dot(los_error, u1); dot(los_error, u2)];  % 2x1

    % Scale by uncertainty
    r_range = w_range * range_error;
    r_los = w_los * los_proj;

    r_i = [r_los; r_range];  % 3-vector: 2 LOS + 1 range

    r(3*(i-1)+1 : 3*i) = r_i;
end

end
