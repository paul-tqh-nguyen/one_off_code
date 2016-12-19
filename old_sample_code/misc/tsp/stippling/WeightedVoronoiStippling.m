function [J] = WeightedVoronoiStippling(I, numGeneratingPoints)

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

[~, C] = kmeans(X,numGeneratingPoints,'start','sample','emptyaction','singleton');

for i = 1:size(C,1)
    fprintf('.')
    J( round(C(i,1)), round(C(i,2)) ) = 1;
end

end