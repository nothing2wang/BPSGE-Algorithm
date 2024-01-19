function [ Aout, xt, error, time ] = SNMF_BPSG_SAGA(y,sr,n_epochs, tau01,tau02, r, Ain, xin) 
% Implement BPSG-SARAH for sparse non-negative matrix factorization
%      argmin_{A,X} \|Y - AX\|_F^2 
%      s.t. \|A_k\|_0 <= tau \|X_k\|_0 <= tau \forall k, A_{i,j} >=0,  X_{i,j} >= 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[n , d] =   size(y);
xi     =    zeros(r,d); 
m       =  floor( n / sr ); 
m2 = floor( d/sr );
grad_book   =   zeros(r , d, sr);
avg         =   sum(grad_book,3)./sr; % creat the average of the gradients
grad_book2   =   zeros(n , r, sr);
avg2         =   sum(grad_book2,3)./sr; % creat the average of the gradients
his         =  zeros(r, d,n_epochs);
error = zeros(n_epochs,1);
pn = 5; 
A = Ain; 
xi = xin;
A_old = A;
xi_old = xi;
t=1;
u_old=100;
uy_old=100;
norm_y = norm(y,'fro');
md = zeros(1,r);
time = zeros(n_epochs, 1);
t_total = 0;
e0 = 0.5 * ( norm( A_old * xi_old - y ,'fro') )^2 ;
for k = 1 : n_epochs 
    tic;    
    idxb  =  randperm(sr,sr);
    idxb2 =  randperm(sr,sr);
    t=t+1;
    for i = 1 : sr         
        if idxb(i) == sr            
         idx    =  (1 + (idxb(i)-1)*m): n; 
        else        
         idx    =  (1 + (idxb(i)-1)*m): (idxb(i)*m);         
        end                    
        As     =   A_old(idx,:);    
        ys     =   y(idx,:);      
        L_A = power_method(As, pn);   
        u  = 1/L_A;
        u = min(u_old, u);
        grad   =   As'*(As*xi_old - ys);        
        grad_diff = grad - grad_book(:, :,idxb(i)); 
        coeff = 3*(norm(A_old,'fro')^2+norm(xi_old,'fro')^2)+norm_y;
        if k > 1      
           xi = coeff*xi_old-(grad_diff + avg)*u;
        else
           xi = coeff*xi_old-grad*u;
        end  
        xi(xi < 0) = 0; 
        xi=xi';
        B = sort(abs(xi), 1, 'descend');
        md = B(tau01,:);
        for q = 1:r 
            xi(:,q) = wthresh(xi(:,q),'h',md(q));
        end
        xi=xi';
              
        avg    =  avg + grad_diff./sr;                
        grad_book(:,:,idxb(i)) = grad;                            
        if idxb2(i) == sr           
         idx    =  (1 + (idxb2(i)-1)*m2): d; 
        else       
         idx    =  (1 + (idxb2(i)-1)*m2): (idxb2(i)*m2);        
        end         
        xi2     =   xi_old(:,idx);
        ys2     =   y(:,idx);       
        L_x = power_method(xi2, pn);     
        uy = 2/L_x;
        uy = min(uy_old, uy);
        grad2   = (xi2*(A*xi2 - ys2)')';  
        grad_diff2 = grad2 - grad_book2(:,:,idxb2(i));        
        if k > 1          
            A = coeff*A_old - (grad_diff2 + avg2)*uy;
        else          
            A = coeff*A_old - grad2*uy;
        end 
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
        xi_old = xi;
        A_old = A;
        avg2    =  avg2 +  grad_diff2./sr;               
        grad_book2(:,:,idxb2(i)) = grad2;            

        uy_old = uy;
        u_old = u;
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







