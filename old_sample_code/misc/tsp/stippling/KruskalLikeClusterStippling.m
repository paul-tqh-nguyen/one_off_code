function [J] = KruskalLikeClusterStippling(I, numGeneratingPoints)

I = im2double(I);

Iheight = size(I,1);
Iwidth = size(I,2);

J = zeros(Iheight, Iwidth);

X=[];

s = GridBasedStippling(I, 10, 1, 1);
[X, Y] = StippleToPoints(s);
X = [X; Y];
X(2,:) = X(2,:)+max(abs(X(2,:)))+1;

I = rgb2gray(I);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

input_points = X;

n = size(input_points,2);
D = dist(input_points,input_points);
clusterLabels = 1:n; % the index refers to which point, the value refers to which tree it belongs to

for i = 1:n % So we don't look at the same edge twice
    for j = 1:i
        D(i,j)=Inf;
    end
end

sortedDistances = sort(unique(D(:)));

for i = 1:numel(sortedDistances)
    distance = sortedDistances(i);
    indicesOfElementsWithCurrentShortestDistance = find(D == distance); % linear time
    for j = 1:numel(indicesOfElementsWithCurrentShortestDistance) % linear time
        index = indicesOfElementsWithCurrentShortestDistance(j);
        [nodeA,nodeB]=ind2sub([n n],index);
        if (clusterLabels(nodeA) ~= clusterLabels(nodeB)) % if the trees are different, connect them
            clusterLabels(clusterLabels == clusterLabels(nodeB)) = clusterLabels(nodeA); % takes less time than find(D == distance)
        end
        if ( numel(unique(clusterLabels)) <=numGeneratingPoints )
            break;
        end
    end
    if (numel(unique(clusterLabels)) <=numGeneratingPoints)
        break;
    end
    
    

    C = zeros(2,numel(unique(clusterLabels)));
    count = 1;
    for i = unique(clusterLabels)    
        C(:,count) = mean( X(:,find(clusterLabels==i)) , 2 );
        count = count + 1;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for i = 1:size(C,2)
        fprintf('.')
        J( round(C(1,i)), round(C(2,i)) ) = 1;
    end

    imshow(J);
    1
    

end

%MUSTNOW CONVERT TO C
C = zeros(2,numel(unique(clusterLabels)));
count = 1;
for i = unique(clusterLabels)    
    C(:,count) = mean( X(:,find(clusterLabels==i)) , 2 );
    count = count + 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:size(C,2)
    fprintf('.')
    J( round(C(1,i)), round(C(2,i)) ) = 1;
end

fprintf('\n')

end