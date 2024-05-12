clear; clc; close all;

% 连接CoppeliaSim
sim = remApi('remoteApi');
sim.simxFinish(-1);  % 确保之前的连接已经关闭
clientID = sim.simxStart('127.0.0.1', 19999, true, true, 5000, 5);
if clientID < 0
    error('无法连接到CoppeliaSim服务器');
end
disp('已连接到CoppeliaSim');

% 获取UR5关节句柄
jointHandles = zeros(1, 6);
for i = 1:6
    [~, jointHandles(i)] = sim.simxGetObjectHandle(clientID, ['UR5_joint', num2str(i)], sim.simx_opmode_blocking);
end

% 设置目标位置
s_position = [0.9, -0.225, 0.6];
tool_position = [-1.2, 0.25, 0.4];

% UR5的DH参数
a = [0, 0.425, 0.39225, 0, 0, 0];
alpha = [pi/2, 0, 0, pi/2, -pi/2, 0];
d = [0.0892, 0, 0, 0.10915, 0.09465, 0.0823];
% a = [0, 0.425, 0.39225, 0, 0, 0];
% alpha = [0, pi/2, 0, pi/2, -pi/2, 0];
% d = [0.0892, 0, 0, 0.10915, 0.09465, 0.0823];

% for i = 1:6
%     [res, theta(i)] = sim.simxGetJointPosition(clientID, jointHandles(i), sim.simx_opmode_blocking);
% end
% theta(2) = theta(2) + pi/2;
theta = [0,pi/2,0,0,0,0];

T = ur5_fk(theta, d, a, alpha);
end_effector_pos = T(1:3, 4)

[~, end_effector_handle] = sim.simxGetObjectHandle(clientID, 'UR5_link7', sim.simx_opmode_blocking);
[~, end_effector_pos] = sim.simxGetObjectPosition(clientID, end_effector_handle, -1, sim.simx_opmode_blocking);

% 三次多项式差值路径规划
a = end_effector_pos;
b = 0;
c = - 0.75 * end_effector_pos + 0.75 * tool_position;
d = 0.5 * end_effector_pos - 0.5 * tool_position;

% 时间参数t从0到1的插值
t = linspace(0, 1, 50); % N是轨迹点的数量

% 计算轨迹点
trajectory = zeros(50, 3); % 初始化轨迹矩阵
for i = 1:50
    trajectory(i, :) = a*t(i)^3 + b*t(i)^2 + c*t(i) + d;
end

plot3(trajectory(1, :), trajectory(2, :), trajectory(3, :))
grid on;

for i = 1:50
   jointAngles = ur5_ik(trajectory(i, :));
   for j = 1:length(jointAngles)
       sim.simxSetJointTargetPosition(clientID, jointHandles(j), jointAngles(j), sim.simx_opmode_blocking);
   end
   sim.simxSynchronousTrigger(clientID);
   sim.simxSynchronous(clientID)
   pause(0.1);
end

    
