function [X,Y] = StippleToPoints(I, perturbationAmount)

% The number of pixels in I which are 1 are the points we will be using to
% draw our final TSP artwork.

%{

I = im2double(I);

Iheight = size(I,1);
Iwidth = size(I,2);

X = zeros(1,sum(I(:)));
Y = zeros(1,sum(I(:)));

count = 0;
for row = 1:Iheight
    for column = 1:Iwidth
        if ( I(row,column) == 1 )
            count = count + 1;
            X(count) = column;
            Y(count) = -row; % We'reconverting from image pixel coordinates to Cartesian coordinates
        end
    end
end

%}

if (nargin < 2)
    perturbationAmount = 0;
end

[Y,X] = find(I == 1);
X =   max( X'+(rand(1,numel(X))*2-1)*perturbationAmount,1);
Y = - max( Y'+(rand(1,numel(Y))*2-1)*perturbationAmount,1);

end