function [t] = ur5_t(theta, d, a, alpha)
    t = [cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha) , a*cos(theta);
         sin(theta), cos(theta)*cos(alpha) , -cos(theta)*sin(alpha), a*sin(theta);
         0         , sin(alpha)            , cos(alpha)            , d           ;
         0         , 0                     , 0                     , 1            ];
end

