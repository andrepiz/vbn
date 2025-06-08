classdef pointUniformVel < propagator
    % pointUniformVel: 3D point with uniform velocity
    % State vector x = [position; velocity], 6x1
    
    properties
        x   % 6x1 state vector: [position; velocity]
    end

    methods
        function obj = pointUniformVel(position, velocity)
            % Constructor
            if nargin == 2
                obj.x = [position(:); velocity(:)];
            else
                obj.x = zeros(6, 1);
            end
        end

        function propagate(obj, dt)
            % PROPAGATE: advance state by dt using constant velocity model
            A = [eye(3), dt * eye(3); zeros(3), eye(3)];  % 6x6
            obj.x = A * obj.x;
        end

        function p = getPosition(obj)
            p = obj.x(1:3);
        end

        function v = getVelocity(obj)
            v = obj.x(4:6);
        end

        function set.x(obj, value)
            % Custom setter for x: enforce 6x1 vector
            if ~isvector(value) || numel(value) ~= 6
                error('State vector x must be a 6-element vector: [position; velocity]');
            end
            obj.x = value(:);  % ensure column vector
        end
    end
end
