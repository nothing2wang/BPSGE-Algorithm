function [ Aout, xt, error, time] = SNMF_BPSG_SGD(y,sr,n_epochs, tau01,tau02, r, Ain, xin)
% Implement BPSG-SARAH for sparse non-negative matrix factorization
%      argmin_{A,X} \|Y - AX\|_F^2 
%      s.t. \|A_k\|_0 <= tau \|X_k\|_0 <= tau \forall k, A_{i,j} >=0,  X_{i,j} >= 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[n , d] =   size(y);
xi     =    zeros(r,d); 
m       =  floor( n / sr ); 
m2 = floor( d/sr );
pn = 5;
error = zeros(n_epochs,1);
A_old = Ain;
xi_old = xin;
norm_y = norm(y,'fro');
t = 1;
pp=0.8;
u_old = 100;
uy_old = 100;
time = zeros(n_epochs, 1);
t_total = 0;
e0 = 0.5*(norm(A_old*xi_old-y,'fro'))^2;
md = zeros(1,r);
for k = 1 : n_epochs
    tic;
    t = t+1;
    tpp=t^pp;
    idxb  =  randperm(sr,sr);
    idxb2 =  randperm(sr,sr);
    for i = 1 : sr
        if idxb(i) == sr
            idx    =  (1 + (idxb(i)-1)*m): n;
        else
            idx    =  (1 + (idxb(i)-1)*m): (idxb(i)*m);
        end
        As     =   A_old(idx,:);
        ys     =   y(idx,:);
        L_A = power_method(A_old(idx,:), pn);
        u  = 1/L_A;
        u = min(u_old, u);
        %
        if idxb2(i) == sr
            idx    =  (1 + (idxb2(i)-1)*m2): d;
        else
            idx    =  (1 + (idxb2(i)-1)*m2): (idxb2(i)*m2);
        end
        xi2     =   xi_old(:,idx);
        ys2     =   y(:,idx);
        L_x = power_method(xi_old(:, idx), pn);
        uy = 1/L_x;
        uy = min(uy_old, uy);
        %
        coeff = 3*(norm(A_old,'fro')^2+norm(xi_old,'fro')^2)+norm_y;
        grad01   =   As'*(As*xi_old - ys)*u;
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
        
        grad2   =  (xi2*(A_old*xi2 - ys2)')'*uy-coeff*A_old;
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
    end
    t1 = toc;
    t_total = t_total + t1;
    time(k) = t_total;
    error(k) = 0.5 * ( norm( A * xi - y ,'fro') )^2 ;
end
xt = xi; 
Aout = A;
error = [e0; error];
end









