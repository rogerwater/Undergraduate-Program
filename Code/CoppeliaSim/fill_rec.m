function [rec] = fill_rec(x, y, z, length_x, length_y)
    fx = [x; x + length_x; x + length_x; x];
    fy = [y; y; y + length_y; y + length_y];
    fz = [z; z; z; z];
    
    rec = fill3(fx, fy, fz, 'r');hold on
end

