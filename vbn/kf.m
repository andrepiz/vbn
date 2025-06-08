classdef kf < handle & matlab.mixin.Copyable
    %KF Kalman Filter
    %   Detailed explanation goes here
    
    properties
        F           % State transition matrix
        H           % Measurement matrix
        P           % State covariance matrix
        Q           % Process covariance matrix
        R           % Measurement covariance matrix
        Q_scaler    % Scaling factor for process noise
        R_scaler    % Scaling factor for measurement noise
    end
    
    methods

    end
end

