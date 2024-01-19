function [ Aout, xt, error, time] = SNMF_BPG(y,n_epochs, tau01,tau02, r, Ain, xin)
% Implement BPSG-SARAH for sparse non-negative matrix factorization
%      argmin_{A,X} \|Y - AX\|_F^2 
%      s.t. \|A_k\|_0 <= tau \|X_k\|_0 <= tau \forall k, A_{i,j} >=0,  X_{i,j} >= 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[n , d] =   size(y);
error = zeros(n_epochs,1);
A_old = Ain;
pn=5;
xi_old = xin;
norm_y = norm(y,'fro');
t = 1;
pp=0.9;
time = zeros(n_epochs, 1);
t_total = 0;
e0 = 0.5 * ( norm( A_old * xi_old - y ,'fro') )^2;
md = zeros(1,r);
u_old=10000;
uy_old=10000;
for k = 1 : n_epochs
    tic;
    t=t+1;
     tpp=t^pp;
    L_A = power_method(A_old, pn);
    u  = 1/L_A;
    u = min(u_old, u);
    coeff = 3*(norm(A_old,'fro')^2+norm(xi_old,'fro')^2)+norm_y;
    grad01   =   A_old'*(A_old*xi_old - y)*u;
    grad = grad01 -coeff*xi_old;
    xi = -grad;
    xi(xi < 0) = 0;
    xi = xi';
    B = sort(abs(xi), 1, 'descend');
    md = B(tau01,:);
    for q = 1:r 
        xi(:,q) = wthresh(xi(:,q),'h',md(q));
    end
    xi=xi';
    

    L_x = power_method(xi_old, pn);   
    uy = 1/L_x;
    
    uy = min(uy_old, uy);
    grad2   =  (xi_old*(A_old*xi_old - y)')'*uy-coeff*A_old;
    A = -grad2;
    A(A<0) = 0;
    B = sort(abs(A), 1, 'descend');
    md = B(tau02,:);
    for q = 1:r 
        A(:,q) = wthresh(A(:,q),'h',md(q));
    end
    
    xi_norm = norm(xi,'fro')^2;
    cor_r_3 = 3*(norm(A,'fro')^2+xi_norm);
    r_sol = solve_eq_3(cor_r_3, 0, norm_y, -1);
    xi = r_sol*xi;
    A = r_sol*A;
    xi_old = xi;
    A_old = A;
    u_old = u;
    uy_old = uy;
    t1 = toc;
    t_total = t_total + 2.1*t1;
    time(k) = t_total;
    error(k) = 0.5 * ( norm( A * xi - y ,'fro') )^2 ;
end
Aout = A;
xt = xi;
error = [e0; error];
end