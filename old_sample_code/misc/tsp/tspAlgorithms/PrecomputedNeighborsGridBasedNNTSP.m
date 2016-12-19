function [output,n] = PrecomputedNeighborsGridBasedNNTSP(stipple, numRowsOfTiles, numColumnsOfTiles)

% Nearest Neighnor TSP
% NNTSP as decribed in CH 6.1.3 Precomputed Neighbors
% We're here computing the NNs of only a subgraph at a time. This reduces
% how much time we have to spend searching ALL of the points in the whole
% graph. We now only have to search through our smaller graph.
% Here, we're going to divide the graph into a 
% numRowsOfTilesxnumOfColumnsOfTiles grid of regions and do a simepl NNTSP
% algorithm on those without connecting the endpoints. We'll then connect
% the end points at the end to get a super nice graph

% O(n^2) where n is the number of points

%fprintf('\nTSP Processing via Grid-Based Precomputed Neighbors Started.\n');
%tic; 

[X,Y] = StippleToPoints(stipple); 
input_points = [X;Y]; % This variable isn't used except to count total number of cities
n = size(input_points,2);

stippleHeight = size(stipple,1);
stippleWidth = size(stipple,2);

horzDim = floor(stippleHeight/numRowsOfTiles);
vertDim = floor(stippleWidth/numColumnsOfTiles);

subTSPGraphs = cell(1,numRowsOfTiles*numColumnsOfTiles);

for rowOfTile = 1:numRowsOfTiles
    if (rowOfTile == numRowsOfTiles)
        rowRange = (1+(rowOfTile-1)*horzDim:stippleHeight);
    else
        rowRange = (1+(rowOfTile-1)*horzDim:rowOfTile*horzDim);
    end
    for columnOfTile = 1:numColumnsOfTiles
        if (columnOfTile == numColumnsOfTiles)
            columnRange = (1+(columnOfTile-1)*vertDim:stippleWidth);
        else
            columnRange = (1+(columnOfTile-1)*vertDim:columnOfTile*vertDim);
        end
        
        % stippleSubTile is the same size as stipple, but removes
        % everythign except for the small portion we are interested in.
        stippleSubTile = zeros(size(stipple));
        stippleSubTile(rowRange,columnRange) = stipple(rowRange,columnRange);
        subTSPGraphs((rowOfTile-1)*numColumnsOfTiles+columnOfTile) = {BasicNNTSP(stippleSubTile)};
    end
end

subTSPGraphs = subTSPGraphs(~cellfun('isempty',subTSPGraphs));
output = subTSPGraphs{1};
potentialPoints = zeros(2,numel(subTSPGraphs)*2);
potentialPoints(:,1) = [Inf; Inf];
potentialPoints(:,2) = [Inf; Inf];
for i = 2:numel(subTSPGraphs)
    currentSubGraph = subTSPGraphs{i};
    potentialPoints(:,(i-1)*2+1) = currentSubGraph(:,1); % odd points are start of subgraph
    potentialPoints(:,(i)*2) = currentSubGraph(:,end); % even points are end of subgraph
end

for i = 1:size(potentialPoints,2)/2
    currentPoint = output(:,end);
    [~, minIndex] = min(dist(currentPoint,potentialPoints));
    if (mod(minIndex,2) == 1)
        whichSubTSPGraph = ceil(minIndex/2);
        output = [output, subTSPGraphs{whichSubTSPGraph}];
        potentialPoints(:,minIndex+1) = [Inf; Inf];
        potentialPoints(:,minIndex) = [Inf; Inf];
    else
        whichSubTSPGraph = minIndex/2;
        output = [output, subTSPGraphs{whichSubTSPGraph}];
        potentialPoints(:,minIndex) = [Inf; Inf];
        potentialPoints(:,minIndex-1) = [Inf; Inf];
    end
end

%output = [output, output(:,1)];

%timePassed = toc;
%fprintf('\nTSP Processing via Grid-Based Precomputed Neighbors finished in %f seconds.\n', timePassed);

end
