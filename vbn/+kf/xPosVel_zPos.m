classdef xPosVel_zPos < kf
    %xPosVel_zPos Kalman Filter for tracking a 3D point
    % position and velocity with position measurements.
    
    properties
        % Tracker state variables
        id          % tracker's id 
        
        % Kalman filter state and parameters
        x     % 6×1 state vector: [x; y; z; vx; vy; vz]
        dt          % fixed time interval
    end
    
    methods
        function self = xPosVel_zPos(x0, P0, dt)
            % Constructor: initialize tracker fields
            
            %-----------------------------
            % Initialize tracker variables
            %-----------------------------
            self.id       = 0;
            
            %-----------------------------
            % Initialize Kalman filter
            %-----------------------------
            % State is [x; y; z; vx; vy; vz]
            self.x = x0;
            self.P = P0;
            self.dt = dt;   % time interval
            
            % State transition matrix (constant velocity model)
            self.F = [ 1, 0, 0, self.dt,    0,       0;
                       0, 1, 0,    0,    self.dt,    0;
                       0, 0, 1,    0,       0,    self.dt;
                       0, 0, 0,    1,       0,       0;
                       0, 0, 0,    0,       1,       0;
                       0, 0, 0,    0,       0,       1 ];
            
            % Measurement matrix: we measure position [x; y; z] only
            self.H = [ 1, 0, 0, 0, 0, 0;
                       0, 1, 0, 0, 0, 0;
                       0, 0, 1, 0, 0, 0 ];
            
            % Initial state covariance
            self.R_scaler = 1.0;
            
            % Process noise covariance (assuming constant‐acceleration noise)
            t2 = self.dt^2;
            t3 = self.dt^3;
            t4 = self.dt^4;
            self.Q = [ t4/4,   0,      0,   t3/2,    0,      0;
                        0,   t4/4,    0,     0,   t3/2,     0;
                        0,     0,   t4/4,    0,      0,   t3/2;
                      t3/2,   0,      0,   t2,      0,      0;
                        0,  t3/2,    0,     0,     t2,      0;
                        0,     0,  t3/2,    0,      0,     t2 ];
            
            % Measurement noise covariance
            self = self.update_R();
        end
        
        function self = update_R(self)
            r_diag = self.R_scaler .* [1; 1; 1];
            self.R = diag(r_diag);
        end
        
        function self = predict_and_update(self, x_meas)
            %PREDICT_AND_UPDATE Perform one Kalman predict‐update cycle
            %   x_meas is a 3×1 measurement [x; y; z]
            
            %-------------------
            % Predict step
            %-------------------
            % Predicted state
            x_pred = self.F * self.x;
            % Predicted covariance
            self.P = self.F * self.P * self.F' + self.Q;
            
            %-------------------
            % Update step
            %-------------------
            % Innovation covariance
            S = self.H * self.P * self.H' + self.R;
            % Kalman gain
            K = self.P * self.H' / S;
            % Residual (innovation)
            y = x_meas - self.H * x_pred;
            % Updated state estimate
            x_new = x_pred + K * y;
            % Updated covariance estimate
            self.P = (eye(6) - K * self.H) * self.P;
            
            % Store updated state
            self.x = x_new;
        end
        
        function self = predict_only(self)
            %PREDICT_ONLY Perform only the Kalman predict step
            %   Used when there is no measurement update
            
            % Predicted state
            x_pred = self.F * self.x;
            % Predicted covariance
            self.P = self.F * self.P * self.F' + self.Q;
            % Store predicted state
            self.x = x_pred;
        end
    end
end
