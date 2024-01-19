function [ Aout, xt, error, time ] = SNMF_BPSGE_SARAH(y,sr,n_epochs, tau01,tau02, r, Ain, xin)
% Implement BPSG-SARAH for sparse non-negative matrix factorization
%      argmin_{A,X} \|Y - AX\|_F^2 
%      s.t. \|A_k\|_0 <= tau \|X_k\|_0 <= tau \forall k, A_{i,j} >=0,  X_{i,j} >= 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[n , d] =   size(y);
m       =  floor( n / sr ); 
m2 = floor( d/sr );
pn = 5;
norm_y = norm(y,'fro');
error = zeros(n_epochs,1);
A = Ain; 
xi = xin;
A_old = A;
xi_old = xi;
A_old_old = A;
xi_old_old = xi;
A_old_old_old = A;
xi_old_old_old = xi;
u_old = 100;
uy_old= 100;
t = 1;
time = zeros(n_epochs, 1);
t_total = 0;
e0 = 0.5 * norm(A_old*xi_old-y,'fro')^2;
md = zeros(1,r);
for k = 1 : n_epochs 
    tic;
    t=t+1;
    beta = 0.6*(k-1)/(k+2); 
    if k>1
        xi_t = xi + beta*(xi-xi_old);
        xi_old = xi;
        L_A = power_method(A_t, pn)/sr;
        u  = 1/L_A;
        u = min(u_old, u);
        coeff = 3*(norm(A_t,'fro')^2+norm(xi_t,'fro')^2)+norm_y;
        grad01   =   A_t'*(A_t*xi_t - y)/sr;
        grad1 = grad01*u -coeff*xi_old;
        xi =-grad1;
        xi(xi < 0) = 0;
        xi = xi';
        B = sort(abs(xi), 1, 'descend');
        md = B(tau01,:);
        for q = 1:r 
            xi(:,q) = wthresh(xi(:,q),'h',md(q));
        end
        xi=xi';
        
        L_x = power_method(xi_t, pn)/sr;
        uy = 2/L_x;
        uy = min(uy_old, uy);
        grad02 = (A_t*xi_t - y)*xi_t'/sr;
        grad2 = grad02*uy-coeff*A_old;
        A = -grad2;
        A(A<0) = 0;
        B = sort(abs(A), 1, 'descend');
        md = B(tau02,:);
        for q = 1:r 
            A(:,q) = wthresh(A(:,q),'h',md(q));
        end
        
        cor_r_3 = 3*(norm(A,'fro')^2+norm(xi,'fro')^2);
        r_sol = solve_eq_3(cor_r_3, 0, norm_y, -1);
        xi = r_sol*xi;
        A = r_sol*A;
        A_old_old_old = A_old_old;
        xi_old_old_old = xi_old_old;
        A_old_old = A_old;
        xi_old_old = xi_old;
        A_old = A;
        xi_old = xi;
        u_old = u;
        uy_old = uy;
    end
    idxb  =  randperm(sr,sr);
    idxb2 =  randperm(sr,sr);

    for i = 1 : sr  
        if idxb(i) == sr
            idx    =  (1 + (idxb(i)-1)*m): n;
        else
            idx    =  (1 + (idxb(i)-1)*m): (idxb(i)*m); 
        end
        A_tt = A_old_old +beta*(A_old_old-A_old_old_old);
        A_t = A_old+beta*(A_old-A_old_old);
        xi_tt = xi_old_old +beta*(xi_old_old-xi_old_old_old);
        xi_t = xi_old +beta*(xi_old-xi_old_old);
        As     =   A_t(idx,:);
        Ass    =   A_tt(idx, :); 
        ys     =   y(idx,:);
        L_A = power_method(A_t(idx,:), pn);
        u  = 1/L_A;
        u = min(u_old, u);
        if k <= 1
            grad01   =  As'*(As*xi_t - ys); 
        else
            grad01   =  As'*(As*xi_t - ys) - Ass'*(Ass*xi_tt - ys) + grad01;    
        end
        
        if idxb2(i) == sr
            idx    =  (1 + (idxb2(i)-1)*m2): d;
        else
            idx    =  (1 + (idxb2(i)-1)*m2): (idxb2(i)*m2);
        end
        xi2     =   xi_t(:,idx);
        xii2    = xi_tt(:,idx);
        ys2     =   y(:,idx);
        L_x = power_method(xi_t(:, idx), pn);
        uy = 2/L_x;
        uy = min(uy_old, uy);
        if k <= 1
            grad02   =  (xi2*(A_t*xi2 - ys2)')' ;
        else
            grad02   = (A_t*xi2 - ys2)*xi2' - (A_tt*xii2 - ys2)*xii2' + grad02;
        end
        coeff = 3*(norm(A_t,'fro')^2+norm(xi_t,'fro')^2)+norm_y;   
        grad1 = grad01*u -coeff*xi_t;
        xi =-grad1; 
        xi(xi < 0) = 0;
        xi = xi';
        B = sort(abs(xi), 1, 'descend');
        md = B(tau01,:);
        for q = 1:r 
            xi(:,q) = wthresh(xi(:,q),'h',md(q));
        end
        xi=xi';
       
        
        grad2 = grad02*uy-coeff*A_t;
        A = -grad2;
        A(A<0) = 0;
        B = sort(abs(A), 1, 'descend');
        md = B(tau02,:);
        for q = 1:r 
            A(:,q) = wthresh(A(:,q),'h',md(q));
        end
       
        cor_r_3 = 3*(norm(A,'fro')^2+norm(xi,'fro')^2);
        r_sol = solve_eq_3(cor_r_3, 0, norm_y, -1);
        xi = r_sol*xi;
        A = r_sol*A;
        A_old_old_old = A_old_old;
        xi_old_old_old = xi_old_old;
        A_old_old= A_old;
        xi_old_old = xi_old;
        A_old = A;
        xi_old = xi;
        u_old = u;
        uy_old = uy;
    end

    t1 = toc;
    t_total = t_total + t1;
    time(k) = t_total;
    error(k) = 0.5 * ( norm( A * xi - y ,'fro') )^2;
end
xt = xi; 
Aout = A;
error = [e0; error];
end











