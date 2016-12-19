

function [D] = dist(A,B)

% Computes the distance of each column of A to each column of B.
% A has m points
% B has n points
% D is m x n
% Returns squared Euclidean distance.

m = size(A,2);
n = size(B,2);
D = zeros(m,n);

for i = 1:m
    D(i,:) = sum((repmat(A(:,i),[1 n]) - B).^2);
end

end