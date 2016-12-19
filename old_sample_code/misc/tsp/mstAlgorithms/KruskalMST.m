function [E] = KruskalMST(stipple)

tic;

if (sum(stipple(:)))
    
    %h=figure;
    
    [X,Y] = StippleToPoints(stipple); 
    input_points = [X;Y];
    
    %input_points = randi(100,2,900);
    
    n = size(input_points,2);
    E = Inf(2,2*n-2);
    D = dist(input_points,input_points); % D is size n x n
    % WE CAN SLIGHTLY PERTURB THIS DATA TO GET RID OF THE SLOW FIND STATEMENT
    treeLabels = 1:n; % the index refers to which point, the value refers to which tree it belongs to
    
    whichEdge = 1;
    sortedDistances = sort(unique(D(:)));
    
    for i = 1:n % So we don't look at the same edge twice
        for j = 1:i
            D(i,j)=Inf;
        end
    end
    
    for i = 1:numel(sortedDistances)
        distance = sortedDistances(i);
        indicesOfElementsWithCurrentShortestDistance = find(D == distance); % linear time
        for j = 1:numel(indicesOfElementsWithCurrentShortestDistance) % linear time
            index = indicesOfElementsWithCurrentShortestDistance(j);
            [nodeA,nodeB]=ind2sub([n n],index);
            if (treeLabels(nodeA) ~= treeLabels(nodeB)) % if the trees are different, connect them
                E(:,2*(whichEdge-1)+1) = input_points(:,nodeA);
                E(:,2*whichEdge) = input_points(:,nodeB);
                whichEdge = whichEdge + 1;
                treeLabels(treeLabels == treeLabels(nodeB)) = treeLabels(nodeA); % takes less time than find(D == distance)
            end
        end
    end
else
    E = [];
end

timePassed = toc;
fprintf('\nKruskalMST finished in %f seconds.\n', timePassed);

end
