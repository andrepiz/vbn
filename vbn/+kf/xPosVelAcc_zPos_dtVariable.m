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
            Q11 = self.dt^5 / 20;
            Q12 = self.dt^4 / 8;
            Q13 = self.dt^3 / 6;
            Q22 = self.dt^3 / 3;
            Q23 = self.dt^2 / 2;
            Q33 = self.dt;

            Qsc = self.Q_scaler;
            if numel(Qsc) == 1
                self.Q = Qsc*[Q11*eye(3), Q12*eye(3), Q13*eye(3);
                              Q12*eye(3), Q22*eye(3), Q23*eye(3);
                              Q13*eye(3), Q23*eye(3), Q33*eye(3)];
            elseif numel(Qsc) == 3
                self.Q = [Qsc(1)*Q11*Qsc(1)*eye(3), Qsc(1)*Q12*Qsc(2)*eye(3), Qsc(1)*Q13*Qsc(3)*eye(3);
                          Qsc(2)*Q12*Qsc(1)*eye(3), Qsc(2)*Q22*Qsc(2)*eye(3), Qsc(2)*Q23*Qsc(3)*eye(3);
                          Qsc(3)*Q13*Qsc(1)*eye(3), Qsc(3)*Q23*Qsc(2)*eye(3), Qsc(3)*Q33*Qsc(3)*eye(3)];
            elseif numel(Qsc) == 9
                self.Q = [Qsc(1)*Q11*eye(3), Qsc(4)*Q12*eye(3), Qsc(7)*Q13*eye(3);
                          Qsc(2)*Q12*eye(3), Qsc(5)*Q22*eye(3), Qsc(8)*Q23*eye(3);
                          Qsc(3)*Q13*eye(3), Qsc(6)*Q23*eye(3), Qsc(9)*Q33*eye(3)];
            else
                error('Q_scaler should be size 1, 3, 3x3 or 9')
            end

        end

        function self = update_R(self)
            % Measurement noise scaler

            Rsc = self.R_scaler;
            if numel(Rsc) == 1
                self.R = Rsc * eye(3); 
            elseif numel(Rsc) == 3
                self.R = diag(Rsc);
            elseif numel(Rsc) == 9
                self.R = reshape(Rsc, 3, 3);
            else
                error('R_scaler should be size 1, 3, 3x3 or 9')
            end
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
