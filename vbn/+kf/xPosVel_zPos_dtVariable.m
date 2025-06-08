classdef xPosVel_zPos_dtVariable < kf
    %xPosVel_zPos_dtVariable Kalman Filter for tracking a 3D point
    % position and velocity with position measurements and variable timestep.
    
    properties
        id          % Tracker ID
        
        x_state     % 6x1 state vector: [x; y; z; vx; vy; vz]
        dt          % Variable time interval
    end
    
    methods
        function self = xPosVel_zPos_dtVariable(x0, P0, dt0)
            % Constructor
            
            % Tracker initialization
            self.id = 0;

            % Kalman Filter initialization
            self.x_state = x0;  % Initial state
            self.P = P0;
            self.dt = dt0;
            
            % Measurement matrix: direct 3D measurement
            self.H = [1 0 0 0 0 0;
                      0 1 0 0 0 0;
                      0 0 1 0 0 0];
            
            % Initial covariance matrices
            self.R_scaler = 1.0;

            % Initialize matrices that depend on dt
            self = self.update_F();
            self = self.update_Q();
            self = self.update_R();
        end
        
        function self = update_F(self)
            % Update state transition matrix based on current dt
            self.F = [ 1, 0, 0, self.dt,    0,       0;
                       0, 1, 0,    0,    self.dt,    0;
                       0, 0, 1,    0,       0,    self.dt;
                       0, 0, 0,    1,       0,       0;
                       0, 0, 0,    0,       1,       0;
                       0, 0, 0,    0,       0,       1 ];
        end
        
        function self = update_Q(self)
            % Update process covariance matrix based on current dt
            dt2 = self.dt^2;
            dt3 = self.dt^3;
            dt4 = self.dt^4;

            self.Q = [dt4/4  0      0     dt3/2  0      0;
                      0     dt4/4  0     0      dt3/2  0;
                      0     0     dt4/4  0      0     dt3/2;
                      dt3/2  0      0     dt2    0      0;
                      0     dt3/2  0     0      dt2    0;
                      0     0     dt3/2  0      0     dt2 ];
        end
        
        function self = update_R(self)
            % Update measurement noise covariance
            r_diag = self.R_scaler .* [1; 1; 1];
            self.R = diag(r_diag);
        end
        
        function self = predict_and_update(self, x_meas, dt)
            % Perform predict and update steps of Kalman filter
            
            self.dt = dt;
            self = self.update_F();
            self = self.update_Q();

            % Predict
            x_pred = self.F * self.x_state;
            self.P = self.F * self.P * self.F' + self.Q;

            % Update
            S = self.H * self.P * self.H' + self.R;
            K = self.P * self.H' / S;
            y = x_meas - self.H * x_pred;
            self.x_state = x_pred + K * y;
            self.P = (eye(6) - K * self.H) * self.P;
        end
        
        function self = predict_only(self, dt)
            % Perform only the predict step (no measurement available)

            self.dt = dt;
            self = self.update_F();
            self = self.update_Q();

            self.x_state = self.F * self.x_state;
            self.P = self.F * self.P * self.F' + self.Q;
        end
    end
end
