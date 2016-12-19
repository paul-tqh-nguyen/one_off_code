function [output,n] = BasicNNTSP(stipple)

if (sum(stipple(:)))
    % Nearest Neighnor TSP
    % Basic NNTSP as decribed in CH 6 of 
    % [Reinelt, 1994] The Traveling Salesman - Computational Solutions for TSP Applications.pdf
    % O(n^2) where n is the number of points
    [X,Y] = StippleToPoints(stipple,1); 
    input_points = [X;Y];
    %size([X;Y])
    
    % n is number of input points
    n = size(input_points,2);
    
    % D is a distance matrix of size n x n
    D = dist(input_points,input_points);

    % points still unconnected
    unconnected = 1:n;

    % which point in input we want to start at
    startPoint = randi(n);
    currentPoint = startPoint;
    unconnected = unconnected(unconnected ~= currentPoint);

    % order is the order in which we visit the cities
    order = zeros(1,n);
    order(:,1) = currentPoint;
    
    for i = 2:n
        %iterationStartTime = toc;
        [~, minIndices] = sort(D(currentPoint,:));
        for j = 1:n
            any(minIndices(j)==unconnected);
            if ( any(minIndices(j)==unconnected) ) % if the closest point is still unconnected
                currentPoint = minIndices(j); % this is our new current point
                order(i) = currentPoint; % add the point to our list of cities to hit
                unconnected = unconnected(unconnected ~= currentPoint); % mark the point we just connected as connected
                break;
            end
        end
        %toc - iterationStartTime
    end
    
    output = zeros(2,n);

    for i = 1:n
        output(:,i) = input_points(:,order(i));
    end
    
    % Reconnect to beginning
    % output(:, n+1) = input_points(:, order(1));
else
    output = [];
end
    
end