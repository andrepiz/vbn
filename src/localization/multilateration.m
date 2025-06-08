function rr_origin2observer = multilateration(ranges, rr_origin2beacons)
% Multilateration: Estimate 3D observer position from ranges to beacons 
% using linear least squares
%
% Inputs:
%   ranges  - 1xN vector, distances from observer to each beacon
%   rr_origin2beacons - 3xN matrix, each column is [x; y; z] of a beacon
%
% Output:
%   rr_origin2observer     - 1x3 vector [x, y, z] estimated observer position
%
%   Author: Andrea Pizzetti
%   Date:   June 6, 2025

[dim, N] = size(rr_origin2beacons);
if dim ~= 3
    error('rr_origin2beacons must be a 3xN matrix');
end
if length(ranges) ~= N
    error('ranges must be a 1xN vector matching rr_origin2beacons');
end
if N < 4
    error('At least 4 rr_origin2beacons are required');
end

% Reference beacon (first column)
p1 = rr_origin2beacons(:, 1);
r1 = ranges(1);

% Build linear system: (N-1)x3 matrix A and (N-1)x1 vector D
A = zeros(N-1, 3);
D = zeros(N-1, 1);

for i = 2:N
    pi = rr_origin2beacons(:, i);
    ri = ranges(i);

    A(i-1, :) = (pi - p1)';  % row vector

    D(i-1) = 0.5 * (r1^2 - ri^2 + sum(pi.^2) - sum(p1.^2));
end

% Check conditioning of A
condA = cond(A);

threshold = 1e4; % conditioning threshold (tunable)
if condA > threshold
    % warning('Matrix A is ill-conditioned (cond = %.2e). Using Tikhonov regularization.', condA);
    % lambda = 1e-4; % regularization parameter (small positive scalar)
    % rr_origin2observer = (A' * A + lambda * eye(3)) \ (A' * D);
    warning('Matrix A is ill-conditioned (cond = %.2e). Rejecting solution.', condA);
    rr_origin2observer = nan(3,1);
else
    rr_origin2observer = A \ D;
end


end
