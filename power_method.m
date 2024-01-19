function [ sig, v ] = power_method( A, nIter )
%       Power Method 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[n, d] = size(A);
xi     =  randn(d, 1);
xi     =  xi./norm(xi);

for  i  =  1 : nIter  
    
    x_tmp  =  A'*(A*xi);
    xi     =  x_tmp./norm(x_tmp);
        
end

v   =  xi;
sig = norm(A'*(A * v));
end

