function rr_origin2observer = lateration(ranges, rr_origin2beacons)
% Lateration: Estimate 3D observer position from ranges to beacons 
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
if N < 3
    error('At least 3 rr_origin2beacons are required');
elseif N == 3
    rr_origin2observer = trilateration(ranges, rr_origin2beacons);
else
    rr_origin2observer = multilateration(ranges, rr_origin2beacons);
end

end
