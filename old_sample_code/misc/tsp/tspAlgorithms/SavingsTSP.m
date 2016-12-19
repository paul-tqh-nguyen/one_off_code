function [output,n] = SavingsTSP(stipple)

if (sum(stipple(:)))
    % Savings TSP
    % Savings TSP Heuristic as decribed in CH 6 of 
    % [Reinelt, 1994] The Traveling Salesman - Computational Solutions for TSP Applications.pdf
    % on page 94
    % O(n^3) bc for every tour (we have n of them), we must check for every
    % pair of tours (which is n choose 2, but in big O that's n^2) which 
    % combination would give us the best savings
    [X,Y] = StippleToPoints(stipple); 
    input_points = [X;Y];
    input_points = randi([0 20],2,25); 
    
    % n is number of input points
    n = size(input_points,2);
    
    % D is a distance matrix of size n x n
    D = dist(input_points,input_points);
    
    % We start with n-1 tours since each tour starts with our base node
    % first row is the tours
    % second row is the tour total distance
    tours = cell(2,n-1); 
    
    for i = 2:n % init our tours
        tours{1,i-1} = [1 i 1]; % 1 is our base node
        tours{2,i-1} = 2 * D(1, i); % second row is the total distance of a tour
    end
    
    for i = 1:n-2 % for all tours - 1
        overallBestToursToMerge = [0 0];
        overallBestNodesToConnectInOverallBestToursToMerge = [-1,-1];
        overallBestShortestDitsance = Inf;
        % for every pair of tours
        for j = 1:size(tours,2) % first tour
            for k = 1:size(tours,2) % second tour
                if (j == k)
                    continue;
                end
                % a,b,c,d are the nodes connected to the base node in our two tours
                a = tours{1,j}(2);
                b = tours{1,j}(end-1);
                c = tours{1,k}(end-1);
                d = tours{1,k}(2);
                % Four choices for connection: ad, ac, bc, bd
                ad_dist = tours{2,j}+tours{2,k}-D(a,1)-D(d,1)+D(a,d);
                ac_dist = tours{2,j}+tours{2,k}-D(a,1)-D(c,1)+D(a,c);
                bd_dist = tours{2,j}+tours{2,k}-D(b,1)-D(d,1)+D(b,d);
                bc_dist = tours{2,j}+tours{2,k}-D(b,1)-D(c,1)+D(b,c);
                [currentBestShortestDitsance, whichPair] = min([ad_dist, ac_dist, bd_dist, bc_dist]);
                if (overallBestShortestDitsance > currentBestShortestDitsance)
                    overallBestShortestDitsance = currentBestShortestDitsance;
                    overallBestToursToMerge = [j k];
                    switch whichPair
                        case 1 % ad
                            overallBestNodesToConnectInOverallBestToursToMerge = [1 1];
                        case 2 % ac
                            overallBestNodesToConnectInOverallBestToursToMerge = [1 2];
                        case 3 % bd
                            overallBestNodesToConnectInOverallBestToursToMerge = [2 1];
                        case 4 % bc
                            overallBestNodesToConnectInOverallBestToursToMerge = [2 2];
                    end
                end
            end
        end
        
        % Merge tours
        t1 = tours{1,overallBestToursToMerge(1)};
        t2 = tours{1,overallBestToursToMerge(2)};
        if (overallBestNodesToConnectInOverallBestToursToMerge(1) == 2)
            t1 = fliplr(t1);
        end
        if (overallBestNodesToConnectInOverallBestToursToMerge(2) == 2)
            t2 = fliplr(t2);
        end
        tours{1,overallBestToursToMerge(1)} = [t1(1:end-1) t2(2:end)];
        tours{2,overallBestToursToMerge(1)} = overallBestShortestDitsance;
        tours(:,overallBestToursToMerge(2)) = [];
        
        %{
        % Code to display each level current plot
        close all;
        figure
        hold
        for j = 1:size(tours,2)
            t=tours{1,j};
            for k = 1:size(t,2)-1
                plot( input_points(1,[t(k) t(k+1)]), input_points(2,[t(k) t(k+1)]) );
                
                plot( input_points(1,t(k)), input_points(2,t(k)), '*');
                plot( input_points(1,t(k+1)), input_points(2,t(k+1)), '*');
                movegui('east');
            end
        end
        axis([-2 22 -2 22]);
        hold
        %}
    end
    
    %{
    close all;
    figure
    hold
    for j = 1:size(tours,2)
        t=tours{1,j};
        for k = 1:size(t,2)-1
            plot( input_points(1,[t(k) t(k+1)]), input_points(2,[t(k) t(k+1)]) );

            plot( input_points(1,t(k)), input_points(2,t(k)), '*');
            plot( input_points(1,t(k+1)), input_points(2,t(k+1)), '*');
            movegui('east');
        end
    end
    axis([-2 22 -2 22]);
    hold
    %}
    output = zeros(2,n);
    
    order = tours{1,1}; % tours will be size n+1
    
    for i = 1:n
        output(:,i) = input_points(:,order(i));
    end
    
else
    output = [];
end
    
end
