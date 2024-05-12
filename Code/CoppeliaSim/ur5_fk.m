function [T] = ur5_fk(theta, d, a, alpha)
    T1 = ur5_t(theta(1), d(1), a(1), alpha(1));
    T2 = ur5_t(theta(2), d(2), a(2), alpha(2));
    T3 = ur5_t(theta(3), d(3), a(3), alpha(3));
    T4 = ur5_t(theta(4), d(4), a(4), alpha(4));
    T5 = ur5_t(theta(5), d(5), a(5), alpha(5));
    T6 = ur5_t(theta(6), d(6), a(6), alpha(6));
    T = T1*T2*T3*T4*T5*T6;
end

