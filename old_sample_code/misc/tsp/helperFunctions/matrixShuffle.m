function [A_shuffled] = matrixShuffle(A)

A_shuffled = A(:,randperm(size(A,2)));

end