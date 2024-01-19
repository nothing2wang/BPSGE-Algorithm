function [im] = eigenfaces(A)


[d, n] = size(A);

d1 = sqrt(d);

n1 = sqrt(n);

im = zeros(d1*n1, d1*n1);

count = 0;
for i = 1:sqrt(n)
    for j = 1:sqrt(n)
        
        count = count + 1;
        
            im(1+(i-1)*d1: i*d1,1+(j-1)*d1: j*d1 ) = reshape(A(:,count), d1, d1) ./ norm(A(:,count),'fro');
    
    
    end
end


end


