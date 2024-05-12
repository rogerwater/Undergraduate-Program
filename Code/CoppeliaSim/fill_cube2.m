function [y1, y2, y3, y4, y5, y6] = fill_cube2(x, y, z, length)
%%
    % 将立方块分为六个平面，每一个平面分别上色
    % 第一个平面
    fx = [x; x + length; x + length; x];
    fy = [y; y; y; y];
    fz = [z; z; z + length; z + length];
    
    y1 = fill3(fx, fy, fz, 'r');hold on
    
    % 第二个平面
    fx = [x + length; x + length; x + length; x + length];
    fy = [y; y + length; y + length; y];
    fz = [z; z; z + length; z + length];
    
    y2 = fill3(fx, fy, fz, 'r');hold on
    
    %第三个平面
    fx = [x; x + length; x + length; x];
    fy = [y + length; y + length; y + length; y + length];
    fz = [z; z; z + length; z + length];
    
    y3 = fill3(fx, fy, fz, 'r');hold on
    
    % 第四个平面
    fx = [x; x; x; x];
    fy = [y; y + length; y + length; y];
    fz = [z; z; z + length; z + length];
    
    y4 = fill3(fx, fy, fz, 'r');hold on
    
    % 第五个平面
    fx = [x; x + length; x + length; x];
    fy = [y; y; y + length; y + length];
    fz = [z + length; z + length; z + length; z + length];
    
    y5 = fill3(fx, fy, fz, 'r');hold on
    
    % 第六个平面
    fx = [x; x + length; x + length; x];
    fy = [y; y; y + length; y + length];
    fz = [z; z; z; z];
    
    y6 = fill3(fx, fy, fz, 'r');hold on
%     pause(0.05)
   
%     fill3(fx, fy, fz, 'w');hold on
end
