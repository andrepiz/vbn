classdef ekf < handle %& matlab.mixin.Copyable
    %EKF Extended Kalman Filter
    %   Detailed explanation goes here
    
    properties
        F           % State transition matrix
        H           % Jacobian of the measurement function
        P           % State covariance matrix
        Q           % Process covariance matrix
        R           % Measurement covariance matrix
        Q_scaler    % Scaling factor for process noise
        R_scaler    % Scaling factor for measurement noise
    end
    
    methods

    end
end

