function [output,n] = ForgottenNodesNNTSP(stipple, initialDegree, distanceThreshold)

% Nearest Neighnor TSP
% NNTSP as decribed in CH 6.1.5 Insertion of Forgotten Nodes
% We try to reduce the number of forgotten nodes, i.e. the ones that are
% close, but not the closest that gets repeatedly ignored until the end 
% (this causes lots of crossing lines, which is a clear indicator of 
% inoptimality). Each node is assigned a degree. Everytime any of it's
% neighbors get added in (neightbors are determined by the distance 
% threshold parameter), it's degree decreases. If it ever hits zero, we add
% it in immediately. Intuitively, once a node has been ignored enough, we
% stop ignoring it.

% O(n^2) where n is the number of points

[X,Y] = StippleToPoints(stipple); 
input_points = [X;Y];

% n is number of input points
n = size(input_points,2);

% D is a distance matrix of size n x n
D = dist(input_points,input_points);

% degrees of each node
degrees = ones(1,n).*initialDegree;

% points still unconnected
unconnected = 1:n;

% which point in input we want to start at
startPoint = randi(n);
degrees(n) = Inf; % When a node is added, it's degree becomes infinity
currentPoint = startPoint;
unconnected = unconnected(unconnected ~= currentPoint);

% order is the order in which we visit the cities
order = zeros(1,n);
order(1) = currentPoint;

%for i = 2:n
i=2;
while (~isempty(unconnected))
    [~, minIndices] = sort(D(currentPoint,:));
    for j = 1:n
        if ( any(minIndices(j)==unconnected) ) % if the closest point is still unconnected
            currentPoint = minIndices(j); % this is our new current point
            order(i) = currentPoint; % add the point to our list of cities to hit
            unconnected = unconnected(unconnected ~= currentPoint); % mark the point we just connected as connected
            degrees(currentPoint) = Inf;
            i = i + 1;
            % Now, we need to decrement the degrees of it's neighbors
            neighbors = floor(dist(input_points(:,currentPoint),input_points)/distanceThreshold)==0;
            degrees(neighbors) = degrees(neighbors)-1;
            forgottenNodes = find(degrees<1);
            for k = 1:size(forgottenNodes,2)
                currentPoint = forgottenNodes(k);
                order(i) = currentPoint;
                unconnected = unconnected(unconnected ~= currentPoint);
                degrees(currentPoint) = Inf;
                i = i + 1;
            end
            break;
        end
    end
end

output = zeros(2,n);

for i = 1:n
    output(:,i) = input_points(:,order(i));
end

% Reconnect to beginning
% output(:, n+1) = input_points(:, order(1));

end
