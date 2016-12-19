function [Rrgb] = GradientDomainCloning(Boriginal, Foriginal, Moriginal)

Rrgb = Boriginal;

M = rgb2gray(Moriginal);

fprintf('\nWorking on red channel.\n');
Rrgb(:,:,1) = GradientDomainCloningHelper(Boriginal(:,:,1),Foriginal(:,:,1),M);

fprintf('\nWorking on green channel.\n');
Rrgb(:,:,2) = GradientDomainCloningHelper(Boriginal(:,:,2),Foriginal(:,:,2),M);

fprintf('\nWorking on blue channel.\n');
Rrgb(:,:,3) = GradientDomainCloningHelper(Boriginal(:,:,3),Foriginal(:,:,3),M);

end

function [R] = GradientDomainCloningHelper(B, F, M)

% Assuming foreground image F and matte imge M are the same dimensions.

% matte image M is white (one) on the foreground and black (zero) on the
% background

R = B;

Mheight = size(M,1);
Mwidth = size(M,2);

n = uint64(sum(sum(M)));
fprintf('\n%d points to calculate.\n', n);

Fgradient = conv2(F, [0 -1 0; -1 4 -1; 0 -1 0]);
A = eye(n).*4;
u = zeros(n,1);
b = zeros(n,1);
indices = find(M ~= 0); % indices of non-zero elements in M

% numel(indices) == n; % Sanity Check

tic;
for i = 1:n % for all points in foreground
    
    if (toc > 1)
        fprintf('\n%d%% done. Working on point %d out of %d.\n',i*100/n, i, n);
        tic;
    end
    
    [currentPointRow,currentPointColumn] = ind2sub(size(M),indices(i)); %convert index indices(i) back to subscripts
    b(i) = Fgradient(currentPointRow,currentPointColumn);
    
    for j = 1:4 % which offset point we're working on, above, below, left, or right
        
        dx = 0;
        dy = 0;
        
        switch j
            case 1 % above
                dy = -1;
            case 2 % below
                dy = 1;
            case 3 % left
                dx = -1;
            case 4 % right
                dx = 1;
        end
        
        offsetPointRow = currentPointRow+dy;
        offsetPointColumn = currentPointColumn+dx;
        
        if (offsetPointColumn > Mwidth || offsetPointColumn < 1 || offsetPointRow > Mheight || offsetPointRow < 1) % if offset point is out of range
            A(i,i) = A(i,i)-1;
        else
            % convert offset point subscript into index
            offsetPointIndex = sub2ind(size(M), offsetPointRow, offsetPointColumn);
            if ( any(indices(:) == offsetPointIndex) )  % if  offset point index is in indices, then it is in the foreground
                % if it is in the foreground, we put a (-1) in the matrix A
                A(i,indices == offsetPointIndex) = -1;
            else % else it is on the boundary
                % if it is on the boundary, we add it to the b value
                [r,c] = ind2sub(size(M), offsetPointIndex);
                b(i) = b(i) + B(r,c);
            end
        end
        %fprintf('.');
    end
    %fprintf('\n');
end

fprintf('\nCalculating our unknown pixels via pcg.\n\n');

u = pcg(A,b, 1e-6, 99999);

fprintf('\nCalculation of uknown pixels done.\n');

fprintf('\nPasting.\n');

for whichIndex = 1:n % go through all the indicesindices
    [r,c] = ind2sub(size(M), indices(whichIndex));
    R(r,c) = u(whichIndex);
end

fprintf('\nDone.\n');

%}

end
