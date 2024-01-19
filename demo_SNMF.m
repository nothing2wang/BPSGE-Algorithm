%       SNMF experiment on ORL data     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%clear;
close all;
Data = load('ORL64');
A  = Data.fea;
A = A';
[n, d] = size(A);
% normalize the data as suggested
for i = 1:d 
    A(:, i) = A(:, i)./norm(A(:, i));   
end
y = A;
maxNumCompThreads(1);
[n, d] = size(y);
% minibatch subsampling ratio = 1/sr = 1/20 = 5%
sr = 5;
n_epochs = 200; % number of total epochs
tau01 = round(d/2);% sparsity constraint for xi
tau02 = round(n/4); % for A
% number of basis image to be extracted
r  = 25; 

load('init_snmf_orl_s');
 % BPG
[ Aout01, xt01, error01 , time01 ] = SNMF_BPG(y,n_epochs, tau01, tau02, r, Ain, xin);
% BPGE
[ Aout02, xt02, error02 , time02 ] = SNMF_BPGE(y,n_epochs, tau01, tau02, r, Ain, xin);
%BPSG-SGD
 [ Aout03, xt03, error03, time03 ] = SNMF_BPSG_SGD(y,sr,n_epochs, tau01, tau02,  r, Ain, xin);
 %BPSGE-SGD
 [ Aout04, xt04, error04, time04 ] = SNMF_BPSGE_SGD(y,sr,n_epochs, tau01, tau02,  r, Ain, xin);

 % BPSG-SAGA
  [ Aout05, xt05, error05, time05 ] = SNMF_BPSG_SAGA(y,sr,n_epochs, tau01, tau02,  r, Ain, xin);
   % BPSG-SAGA
  [ Aout06, xt06, error06, time06 ] = SNMF_BPSGE_SAGA(y,sr,n_epochs, tau01, tau02,  r, Ain, xin);

% BPSG-SARAH
  [ Aout07, xt07, error07, time07 ] = SNMF_BPSG_SARAH(y,sr,n_epochs, tau01, tau02,  r, Ain, xin);
  % BPSGE-SARAH
  [ Aout08, xt08, error08, time08 ] = SNMF_BPSGE_SARAH(y,sr,n_epochs, tau01, tau02,  r, Ain, xin);

bound = 7777;%2.3;
%%
linewidth = 1;
axesFontSize = 6;
labelFontSize = 11;
legendFontSize = 8;
resolution = 108; 
output_size = resolution *[12, 12]; 

%%%%%% %%%%%% %%%%%% %%%%%% %%%%%%
figure(102), clf;
p1 = plot(0:1:n_epochs, min(bound,log10(error01(1:end))),'x--','LineWidth',1.5, 'color', [0,1,0], 'MarkerIndices', 1:n_epochs/5:n_epochs,'MarkerSize',10);
hold on
p2 = plot(0:1:n_epochs, min(bound,log10(error02(1:end))),'*-','LineWidth',1.5, 'color', [0,1,0], 'MarkerIndices', 1:n_epochs/5:n_epochs,'MarkerSize',10);
hold on
p3 = plot(0:1:n_epochs, min(bound,log10(error03(1:end))), 's--','LineWidth',1.5,'color', [1,0,1], 'MarkerIndices', 1:n_epochs/5:n_epochs,'MarkerSize',10);
hold on
p4 = plot(0:1:n_epochs, min(bound,log10(error04(1:end))), 'o-','LineWidth',1.5,'color', [1,0,1], 'MarkerIndices', 1:n_epochs/5:n_epochs,'MarkerSize',10);
hold on
p5 = plot(0:1:n_epochs, min(bound,log10(error05(1:end))), 'd--','LineWidth',1.5,'color', [0,0,1], 'MarkerIndices', 1:n_epochs/5:n_epochs,'MarkerSize',10);
hold on
p6 = plot(0:1:n_epochs, min(bound,log10(error06(1:end))), 'p-','LineWidth',1.5,'color', [0,0,1], 'MarkerIndices', 1:n_epochs/5:n_epochs,'MarkerSize',10);
hold on
p7 = plot(0:1:n_epochs, min(bound,log10(error07(1:end))), '^--','LineWidth',1.5,'color', [1,0,0], 'MarkerIndices', 1:n_epochs/5:n_epochs,'MarkerSize',10);
hold on
p8 = plot(0:1:n_epochs, min(bound,log10(error08(1:end))), 'v-','LineWidth',1.5,'color', [1,0,0], 'MarkerIndices', 1:n_epochs/5:n_epochs,'MarkerSize',10);
hold off
set(gca,'FontSize', 12);
grid on;
lg = legend([p1, p3, p5, p7, p2,p4, p6, p8],'BPG', 'BPSG-SGD', 'BPSG-SAGA', 'BPSG-SARAH', 'BPGE','BPSGE-SGD', 'BPSGE-SAGA', 'BPSGE-SARAH', 'NumColumns',2);
legend('boxoff');
set(lg, 'Location', 'NorthEast');
set(lg, 'FontSize', 10);
ylb = ylabel({'$\mathrm{log}(\Phi(U_k, V_k))$'},'FontAngle', 'normal', 'Interpreter', 'latex', 'FontSize', 16);
set(ylb, 'Units', 'Normalized', 'Position', [-0.08, 0.5, 0]);
xlb = xlabel({'\# of epochs'}, 'FontSize', 14,'FontAngle', 'normal', 'Interpreter', 'latex');
set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.07, 0]);
set (gcf,'Position',[440,378,560,350])





