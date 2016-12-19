function [J] = KMedoidsStippling(I, numGeneratingPoints)

I = im2double(rgb2gray(I));

Iheight = size(I,1);
Iwidth = size(I,2);

J = zeros(Iheight, Iwidth);

X=[];

for r = 1:Iheight
    %fprintf('\n%d: ',r)
    for c = 1:Iwidth
        %fprintf('%d ',c)
        numPointsToAdd = floor(I(r,c)*10);
        X = [X;repmat([r c],numPointsToAdd, 1)];
    end
end

size(X)

[~,C,~] = kmedoid2(X, numGeneratingPoints, 500);

for i = 1:size(C,1)
    %fprintf('.')
    J( round(C(i,1)), round(C(i,2)) ) = 1;
end

end

% Code below was stolen from somewhere
function [IDX, Cluster, Err] = kmedoid2(data, NC, maxIter, varargin)
% Performs clustering using Partition Around Medoids (PAM) algorithm
% Initial clusters must be selected from the available data points

% Usage
% [IDX,C,cost] = kmedoid(data, NC, maxIter, [init_cluster])
% 
% Input : 
%       data - input data matrix
%       NC - number of clusters
%       maxIter - Maximum number of iterations
%       init_cluster - Initial cluster centers (optional)
%       init_index - Index of initial clusters
%
% Output :
%       IDX - data point index which became cluster centres
%       C -  Cluster centers
%       Err - cost associated with the clusters for all iterations
% ------------------------------------------

% Size of data
dsize = size(data);

% Number of data points
L = dsize(1);

% No. of features
NF = dsize(2); 

% Cluster size
csize = [NC, NF];

if (L < NC)
    error('Too few data points for making k clusters');
end

if(nargin > 5)
    error('Usage : [IDX,C,error] = kmedoid(data, NC, maxIter, [init_cluster], [init_idx])');
elseif(nargin == 5)
    vsize1 = size(varargin{1});
    vsize2 = length(varargin{2});
    
    if(isequal(vsize1,csize))
        Cluster = varargin{1};
    else
        error('Incorrect size for initial cluster');
    end
    
    if(vsize2 == NC)
        IDX = varargin{2};
    else
        error('Incorrect size for initial cluster index');
    end
elseif(nargin == 4)
   error('You must provide two optional arguments: init_cluster, init_idx');
elseif(nargin == 3) % no initial cluster provided
    IDX = randint(NC,1,L)+1;
    Cluster = data(IDX,:); % Initialize the initial clusters randomly
else
    display('Usage : [IDX,C] = kmedoid(data, NC, maxIter, [init_cluster], [init_idx]');
    error('Function takes at least 3 arguments');
end

%Array for storing cost for each iteration
Err = zeros(maxIter,1);

Z = zeros(NC, NF, L);
for i = 1:L
    Z(:,:,i) = repmat(data(i,:),NC,1);
end


FIRST = 1;
total_cost = 0;


for Iteration = 1:maxIter
    %disp(Iteration);
    
    if(FIRST) 
        % First time, compute the cost associated with the initial cluster
        
        C = repmat(Cluster,[1,1,L]);
        B = sqrt(sum((abs(Z-C).^2),2)); %Euclidean distance
        B1 = squeeze(B);
        [Bmin,Bidx] = min(B1,[],1);

        cost = zeros(1,NC);
        for k = 1:NC
            cost(k) = sum(Bmin(Bidx==k));
        end
        total_cost = sum(cost);
        
        FIRST = 0; % Reset the FIRST flag
        
    else % Not first time
        
        % change one cluster center randomly and see if the cost is
        % decreased. If yes, accept the change, otherwise reject the change

        while(1)
            Tidx = randint(1,1,L)+1; % find a random number betn 1 to L
            if(isempty(find(IDX==Tidx))) % Avoid previously selected centers
                break;
            end
        end
        % randomly change one cluster
        pos = randint(1,1,NC) + 1;
        OLD_IDX = IDX; % Preserve the old index list
        IDX(pos) = Tidx;

        %Assign cluster centers
        PCluster = Cluster;
        Cluster = data(IDX,:);

        % For each data point, find the cluster which is closest to it
        % and compute the cost obtained from this association

        C = repmat(Cluster,[1,1,L]);

        %B = sum(abs(Z-C),2); %Manhattan distance
        B = sqrt(sum((abs(Z-C).^2),2)); %Euclidean distance

        % Row - cluster (NC)
        % Column - each data point 1, 2, ... L
        B1 = squeeze(B);

        % For each data point, find the nearest cluster
        % size(Bmin) = 1xL
        [Bmin,Bidx] = min(B1,[],1);

        cost = zeros(1,NC);
        for k = 1:NC
            cost(k) = sum(Bmin(Bidx==k));
        end

        previous_cost = total_cost;
        total_cost = sum(cost);


        if(total_cost > previous_cost) % cost increases
            % Bad choice, restore the previous index list
            IDX = OLD_IDX;
            total_cost = previous_cost;
            Cluster = PCluster;
        end
    end
    Err(Iteration) = total_cost;
end

if(nargout == 0)
    disp(IDX);
elseif(nargout > 3)
    error('Function gives only 3 outputs');
end


end