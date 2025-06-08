function [xyz, R] = uvd2xyz(K, u, v, d, sigma_u, sigma_v, sigma_d)
% Return the 3D vector from the camera pixel coordinates U, V and corresponding
% depth D. 
% Return the 3x3 covariance matrix of a 3D point given the corresponding
% uncertainties

% backprojection
% un = (u - K(1,3))./K(1,1);
% vn = (v - K(2,3))./K(2,2);
% if any(un.^2 + vn.^2 >= 1)
%     error('U and V are outside the resolution limits or K is defined uncorrectly');
% end
% dz = sqrt(1 - un.^2 - vn.^2);
% xyz = d.*[un; vn; dz];

un = (u - K(1,3))./K(1,1);
vn = (v - K(2,3))./K(2,2);
if any(un.^2 + vn.^2 >= 1)
    error('U and V are outside the resolution limits or K is defined uncorrectly');
end
zn = ones(size(un));
rn = sqrt(un.^2 + vn.^2 + zn.^2);
xyz = d.*[un; vn; zn]./rn;

if ~exist('sigma_u','var') || ~exist('sigma_v','var') || ~exist('sigma_d','var')
    R = [];
    return
end

% uncertainties in normalized image coordinates
sigma_un = (sigma_u)./K(1,1);
sigma_vn = (sigma_v)./K(2,2);

% % Jacobian 
% % J = d(rPlSc)/d([un, vn, d])
% % J = δ([d*un; d*vn; d*sqrt(1 - un^2 - vn^2)])/δ([un, vn, d])
% 
% J = [d,          0,          un;
%      0,          d,          vn;
%     -d*un/dz,   -d*vn/dz,    dz];

% Jacobian 
% J = d(rPlSc)/d([un, vn, d])
% J = δ([d*un/rn; d*vn/rn; d/rn])/δ([un, vn, d])

J = [d*(1/rn - un^2/rn^3),          -d*un*vn/rn^3,                  un/rn;
    -d*un*vn/rn^3,                   d*(1/rn - vn^2/rn^3),          vn/rn;
    -d*un/rn^3,                     -d*vn/rn^3,                     1/rn];

% Measurement covariance matrix
Ruvd = diag([sigma_un^2, sigma_vn^2, sigma_d^2]);

% Propagate uncertainty
R = J * Ruvd * J';

%R_N =  NC * R_C * NC'
end