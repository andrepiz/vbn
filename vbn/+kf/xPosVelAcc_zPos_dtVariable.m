classdef xPosVelAcc_zPos_dtVariable < kf
    %xPosVelAcc_zPos_dtVariable Kalman Filter for tracking a 3D point
    % position, velocity and acceleration with position measurements and variable timestep.
    
    properties
        id
        
        x_state    % [x; y; z; vx; vy; vz; ax; ay; az]
        dt
    end

    methods
        function self = xPosVelAcc_zPos_dtVariable(x0, P0, dt0)

            self.id = 0;

            self.x_state = x0;  % 9D state: position, velocity, acceleration
            self.P = P0;
            self.dt = dt0;

            % Measurement matrix assumes direct position measurement
            self.H = [1 0 0 0 0 0 0 0 0;
                      0 1 0 0 0 0 0 0 0;
                      0 0 1 0 0 0 0 0 0];

            self.R_scaler = 1.0;
            self.Q_scaler = 1.0;

            self = self.update_F();
            self = self.update_Q();
            self = self.update_R();
        end

        function self = update_F(self)

            dt2 = self.dt^2 / 2;

            self.F = eye(9);
            % Position to velocity
            self.F(1,4) = self.dt;
            self.F(2,5) = self.dt;
            self.F(3,6) = self.dt;
            % Position to acceleration
            self.F(1,7) = dt2;
            self.F(2,8) = dt2;
            self.F(3,9) = dt2;
            % Velocity to acceleration
            self.F(4,7) = self.dt;
            self.F(5,8) = self.dt;
            self.F(6,9) = self.dt;
        end

        function self = update_Q(self)
            % Process noise: simple model with acceleration noise

            % Elements for process noise covariance matrix (jerk noise model)
            q11 = self.dt^5 / 20;
            q12 = self.dt^4 / 8;
            q13 = self.dt^3 / 6;
            q22 = self.dt^3 / 3;
            q23 = self.dt^2 / 2;
            q33 = self.dt;
            
            Q_block = self.Q_scaler * [q11, q12, q13;
                                        q12, q22, q23;
                                        q13, q23, q33];
            % Block diagonal Q for x,y,z axes
            self.Q = blkdiag(Q_block, Q_block, Q_block);
        end

        function self = update_R(self)
            r_diag = self.R_scaler .* [1; 1; 1];
            self.R = diag(r_diag);
        end

        function self = predict_and_update(self, x_meas, dt)
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
            self.P = (eye(9) - K * self.H) * self.P;
        end

        function self = predict_only(self, dt)
            self.dt = dt;
            self = self.update_F();
            self = self.update_Q();

            self.x_state = self.F * self.x_state;
            self.P = self.F * self.P * self.F' + self.Q;
        end
    end
end
