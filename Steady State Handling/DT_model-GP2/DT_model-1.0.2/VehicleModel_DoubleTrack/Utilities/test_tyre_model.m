%% Test Tyre model

load_MF96_tyre_data;

alpha_swipe = -1:0.01:1;
kappa_swipe = -1:0.01:1;

alpha_vec = [0 10 20 30 50]'*pi/180;
phi_vec = [0 1 2 3 5]'*pi/180;
fz_vec = [2500 3500 4500 5500 6500];
kappa_vec = [0 0.1 0.2 0.5 0.8]';
fz_nom = 4500;


%% Lateral Force
% Pure Fy var Fz
figure
hold on
tmp_var_vec = fz_vec;
for jj = 1:length(tmp_var_vec)
    plot(alpha_swipe,...
        forloop(alpha_swipe,@(alpha)MF96_FY0(0,alpha,0,tmp_var_vec(jj),tyre_data_f)),...
        "DisplayName",strcat("$F_z$ = ",num2str(tmp_var_vec(jj))),...
        "LineWidth",2);
end
hold off
legend
grid
xlabel('$\alpha$ [rad]'); ylabel('$F_y$ [N]')
title('Pure $F_y$ varying $F_z$')

% Pure Fy var Fz
figure
hold on
tmp_var_vec = phi_vec;
for jj = 1:length(tmp_var_vec)
    plot(alpha_swipe,...
        forloop(alpha_swipe,@(alpha)MF96_FY0(0,alpha,tmp_var_vec(jj),fz_nom,tyre_data_f)),...
        "DisplayName",strcat("$\gamma$ = ",num2str(rad2deg(tmp_var_vec(jj)))),...
        "LineWidth",2);
end
hold off
legend
grid
xlabel('$\alpha$ [rad]'); ylabel('$F_y$ [N]')
title('Pure $F_y$ varying $\gamma$')

% Fy var kappa
figure
hold on
tmp_var_vec = kappa_vec;
for jj = 1:length(tmp_var_vec)
    plot(alpha_swipe,...
        forloop(alpha_swipe,@(alpha)MF96_FY(tmp_var_vec(jj),alpha,0,fz_nom,tyre_data_f)),...
        "DisplayName",strcat("$\kappa$ = ",num2str(rad2deg(tmp_var_vec(jj)))),...
        "LineWidth",2);
end
hold off
legend
grid
xlabel('$\alpha$ [rad]'); ylabel('$F_y$ [N]')
title('Combined $F_y$ varying $\kappa$')

%% Longitudinal Force
% Pure Fx var Fz
figure
hold on
tmp_var_vec = fz_vec;
for jj = 1:length(tmp_var_vec)
    plot(kappa_swipe,...
        forloop(kappa_swipe,@(kappa)MF96_FX0(kappa,0,0,tmp_var_vec(jj),tyre_data_f)),...
        "DisplayName",strcat("$F_z$ = ",num2str(tmp_var_vec(jj))),...
        "LineWidth",2);
end
hold off
legend
grid
xlabel('$\kappa$ [-]'); ylabel('$F_x$ [N]')
title('Pure $F_x$ varying $F_z$')

% Pure Fy var Fz
figure
hold on
tmp_var_vec = phi_vec;
for jj = 1:length(tmp_var_vec)
    plot(kappa_swipe,...
        forloop(kappa_swipe,@(kappa)MF96_FX0(kappa,0,tmp_var_vec(jj),fz_nom,tyre_data_f)),...
        "DisplayName",strcat("$\gamma$ = ",num2str(rad2deg(tmp_var_vec(jj)))),...
        "LineWidth",2);
end
hold off
legend
grid
xlabel('$\kappa$ [-]'); ylabel('$F_x$ [N]')
title('Pure $F_x$ varying $\gamma$')

% Fy var kappa
figure
hold on
tmp_var_vec = alpha_vec;
for jj = 1:length(tmp_var_vec)
    plot(kappa_swipe,...
        forloop(kappa_swipe,@(kappa)MF96_FX(kappa,tmp_var_vec(jj),0,fz_nom,tyre_data_f)),...
        "DisplayName",strcat("$\alpha$ = ",num2str(rad2deg(tmp_var_vec(jj)))),...
        "LineWidth",2);
end
hold off
legend
grid
xlabel('$\kappa$ [-]'); ylabel('$F_x$ [N]')
title('Combined $F_x$ varying $\alpha$')










%% Aux functions

function res = forloop(loop_vec,fun_handle)
for k = 1:length(loop_vec)
    res(k,1) = fun_handle(loop_vec(k));
end
end