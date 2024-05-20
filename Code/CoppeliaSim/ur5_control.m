clear; clc; close all;
s_position = [-1, 0.3, 0.25]; % 卫星位置
tool_position = [0.4, -1, 0.25]; % 工具箱位置

%% 使用标准DH参数建立UR5机器人
% [theta, d, a, alpha, sigma]
L(1) = Link([0, 0.08915, 0, pi/2]);
L(2) = Link([0, 0, -0.425, 0]);
L(2).offset = -pi/2;
L(3) = Link([0, 0, -0.39225, 0]);
L(4) = Link([0, 0.10915, 0, pi/2]);
L(5) = Link([0, 0.09465, 0, -pi/2]);
L(6) = Link([0, 0.08230, 0, 0]);
q_lim = [-pi, pi];
for i = 1:6
    L(i).qlim = q_lim;
end
ur5 = SerialLink(L, 'name', 'ur5');

% T = ur5.fkine([0, -pi/2, 0, 0, 0, 0])
% Q = ur5.ikine(T)
% ur5.plot([0, 0, 0, 0, 0, 0]);
% ur5.display;
% ur5.teach;

%% 蒙特卡洛法求UR5机器人工作空间
% N = 5000;
% theta_1 = q_lim + diff(q_lim) * rand(N, 1); % 关节1限制
% theta_2 = q_lim + diff(q_lim) * rand(N, 1); % 关节2限制
% theta_3 = q_lim + diff(q_lim) * rand(N, 1); % 关节3限制
% theta_4 = q_lim + diff(q_lim) * rand(N, 1); % 关节4限制
% theta_5 = q_lim + diff(q_lim) * rand(N, 1); % 关节5限制
% theta_6 = q_lim + diff(q_lim) * rand(N, 1); % 关节6限制
% pos = {1, N};
% % 正运动学求解
% for i = 1:N
%     pos{i} = ur5.fkine([theta_1(i), theta_2(i), theta_3(i), theta_4(i), theta_5(i), theta_6(i)]);
% end
% % 绘图
% figure
% ur5.plot([0, 0, 0, 0, 0, 0], 'jointdiam', 1);
% axis equal;
% hold on;
% for i = 1:N
%     plot3(pos{i}.t(1), pos{i}.t(2), pos{i}.t(3), 'r.');
%     hold on;
% end
% grid off
% view(20, 30)

%% UR5机器人轨迹规划

t = 0:0.05:5;
% UR5 DH参数
a = [0, -0.425, -0.39225, 0, 0, 0];
alpha = [pi/2, 0, 0, pi/2, -pi/2, 0];
d = [0.0892, 0, 0, 0.10915, 0.09465, 0.0823];
theta = [0, -pi/2, 0, 0, 0, 0];
T = ur5_fk(theta, d, a, alpha);
q0 = [0, 0, 0, 0, 0, 0];
q = ur5_ik(T) ;


% UR5从初始位置移动到工具箱处
q1 = [-pi/4, -pi/2, pi/4, pi/2, 0, pi/2];
qc1 = jtraj(q0, q1, t);
T1 = ur5.fkine(q1);
figure(1)
[y01, y02, y03, y04, y05, y06] = fill_cube(-0.1, -0.1, -0.3, 0.2);
[y11, y12, y13, y14, y15, y16] = fill_cube2(s_position(1), s_position(2), s_position(3), 0.45);
[y21, y22, y23, y24, y25, y26] = fill_cube(tool_position(1), tool_position(2), tool_position(3), 0.3);
rec1 = fill_rec(s_position(1) - 0.7, s_position(2), s_position(3) + 0.25, 0.7, 0.3);
rec2 = fill_rec(s_position(1) + 0.4, s_position(2), s_position(3) + 0.25, 0.7, 0.3);
y2 = [];
hold on
ur5.plot(qc1, 'trail', 'b-', 'movie', '1.mp4');

% UR5从工具箱处移动到卫星处
q2 = [0, pi/2, -pi/4, 0, pi/2, 0];
qc2 = jtraj(q1, q2, t);
T2 = ur5.fkine(q2);
ur5.plot(qc2, 'trail', 'b-', 'movie', '2.mp4');

% UR5从卫星处移动到工具箱处
q3 = [-pi/4, -pi/2, pi/4, pi/2, 0, pi/2];
qc3 = jtraj(q2, q3, t);
ur5.plot(qc3, 'trail', 'b-', 'movie', '3.mp4');


