function [E] = PrimMST(stipple)

tic;

%{
This actually takes much longer that TSP 

in the second for loop of tsp, we search through a bunch fo points that are
sorted by closeness. we only need to go through this set of size n until we
find one that has not been added yet. We can then break out of this loop
early. it goes at max m times where m is the number of points yet to have
been connected.

in the second for loop of prim, we need to update distances everytime of
every point not added yet. This always runs m times where m is the number
of points not added yet. We cannot break early bc we must always update the
distance values.
%}

if (sum(stipple(:)))
    % Prim's Algo for MST
    % http://en.wikipedia.org/wiki/Prim%27s_algorithm
    % O(n^2) where n is the number of points
        
    [X,Y] = StippleToPoints(stipple); 
    input_points = [X;Y];
    %input_points = randi(100,2,900);
    
    n = size(input_points,2);
    D = dist(input_points,input_points); % D is size n x n
    edgeIndices = Inf(2,n-1); % top and bottom are indices of the edge
    
    distancesOfNodesFromTree = Inf(2,n-1); % top = dist, bottom = node
    nodesToBeAdded = 1:n; 
    
    [~,minIndex] = min(D(:));
    [node1,node2]=ind2sub([n n],minIndex);
    edgeIndices(:,1) = [node1; node2];
    nodesToBeAdded = nodesToBeAdded(nodesToBeAdded ~= node1);
    nodesToBeAdded = nodesToBeAdded(nodesToBeAdded ~= node2);
    
    for j = 1:n % update distance values
        [minVal, minIndex] = min(D([node1, node2],j));
        distancesOfNodesFromTree(1,j) = minVal;
        if (minIndex == 1)
            distancesOfNodesFromTree(2,j) = node1;
        else
            distancesOfNodesFromTree(2,j) = node2;
        end
    end
    distancesOfNodesFromTree(:,node1) = [Inf;node2];
    distancesOfNodesFromTree(:,node2) = [Inf;node1];
    
    for i = 2:n-1 % for all nodes not yet attached to MST
        %iterationStartTime = toc;
        [~,closestNode] = min(distancesOfNodesFromTree(1,:));
        
        edgeIndices(:,i) = [closestNode; distancesOfNodesFromTree(2,closestNode)];
        
        nodesToBeAdded = nodesToBeAdded(nodesToBeAdded ~= closestNode);
        
        for j = nodesToBeAdded % update distance values
            if ( D(j,closestNode) < distancesOfNodesFromTree(1,j) )
                distancesOfNodesFromTree(:,j) = [D(j,closestNode); closestNode];
            end
        end
        
        distancesOfNodesFromTree(1,closestNode) = Inf; % so we don't pick it again
        %toc - iterationStartTime
    end
    
    % E is a 2 x 2*(n-1) matrix where each consecutive pair of points is an edge
    E = Inf(2,2*n-2);
    
    for i = 1:size(edgeIndices,2)
        E(:,2*(i-1)+1) = input_points(:,edgeIndices(1,i));
        E(:,2*i) = input_points(:,edgeIndices(2,i));
    end
    
else
    E = [];
end

timePassed = toc;
fprintf('\nPrimMST finished in %f seconds.\n', timePassed);

end
