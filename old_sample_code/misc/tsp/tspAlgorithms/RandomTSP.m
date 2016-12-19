function [output,n] = RandomTSP(stipple)

if (sum(stipple(:)))
    
    % TSP tour by just picking random points
    
    [X,Y] = StippleToPoints(stipple); 
    input_points = [X;Y];
    n = size(input_points,2);
    
    output = input_points(:,randperm(size(input_points,2)));
else
    output = [];
end
    
end