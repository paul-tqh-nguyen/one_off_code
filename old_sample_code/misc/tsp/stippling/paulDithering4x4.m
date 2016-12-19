function [J] = paulDithering(I, tileDim, sparsityFactor)

% We're going to implement ordered dithering as described by Kaplan and
% Bosch.

% Paper: http://www.cgl.uwaterloo.ca/~csk/papers/kaplan_bridges2005b.pdf

% Ordered dithering involves us distributing our points according to
% patterns (or lattices as the paper describes it) instead of uniformly
% randomly.

% The paper makes it seem that the lattices each must be predetermined
% by the programmer. The wikipedia article tells us that there are various
% threshold maps available. Thus, we must pick the best. 
% Here, we're going to use the same dithering grids Kaplan and Bosch used
% in their paper. 

% We're going to take each tileDim x tileDim grid and replace it with an
% an approriately matching 4 x 4 dithering grid. 


%{

% Test Script

I = imread('monalisa.jpeg');
Iheight = size(I,1);
Iwidth = size(I,2);

scaleFactor = 10;
tileDim = 10;
sparsityFactor = 1;

J = orderedDitheringKaplanBosch4x4(I, tileDim, sparsityFactor);
output = NNTSP(J); 

Jheight = size(J,1);
Jwidth = size(J,2);

dispayDim = max(Jwidth+1,Jheight+1);

center = [floor(Jwidth/2) floor(-Jheight/2)];
XCOORD = 1;
YCOORD = 2;

plot(output(XCOORD,:),output(YCOORD,:));
padding = min(floor(Iheight/20),floor(Iwidth/20));
axis([center(XCOORD)-(ceil(dispayDim/2)+padding) center(XCOORD)+(ceil(dispayDim/2)+padding) center(YCOORD)-(ceil(dispayDim/2)+padding) center(YCOORD)+(ceil(dispayDim/2)+padding)]);

%}

%fprintf('\nStippling via Ordered Dithering Started.\n');
%tic; 

ditheringGrids = determineDitheringGrid();
ditheringGridDensities = determineDitheringGridDensities(ditheringGrids);
ditheringGridDim = size(ditheringGrids,1);

I = fitImage(im2double(rgb2gray(I)), tileDim);

Iheight = size(I,1);
Iwidth = size(I,2);

numRowsOfTiles = Iheight/tileDim; % number of rows of tiles
numColumnsOfTiles = Iwidth/tileDim; % number of columns of tiles

J = zeros(numRowsOfTiles*ditheringGridDim,numColumnsOfTiles*ditheringGridDim);

for row = 1:numRowsOfTiles
    for column = 1: numColumnsOfTiles
        tile = getTile(row,column,I, tileDim);
        tileDensity = mean(tile(:));
        tileDensity=tileDensity/sparsityFactor; % cut down density by sparsityFactor
        %time to put points down
        % We assign the new grid to be the grid whose density matches that of the tile in question the best
        % We do this by just finding the grid with teh smallest density
        % difference with our tile in question
        [~,newGridIndex] = min(abs(ditheringGridDensities - tileDensity)); 
        newGrid = ditheringGrids(:,:,newGridIndex);
        J(1+(row-1)*ditheringGridDim:row*ditheringGridDim,1+(column-1)*ditheringGridDim:column*ditheringGridDim,:) = newGrid;
    end
end

%timePassed = toc;
%fprintf('\nStippling via Ordered Dithering finished in %f seconds.\n', timePassed);

end

function [ditheringGridDensities] = determineDitheringGridDensities(ditheringGrids)

numDensityGrids = size(ditheringGrids,3);

ditheringGridDensities = zeros(1,numDensityGrids);

for i = 1:numDensityGrids
    grid = ditheringGrids(:,:,i);
    ditheringGridDensities(i) = mean(grid(:));
end

end

function [ditheringGrids] = determineDitheringGrid()

ditheringGrids = zeros(4,4,6);

ditheringGrids(:,:,1) = [ 0 0 0 0  ; ...
                          0 0 0 0  ; ...
                          0 0 0 0  ; ...
                          0 0 0 0 ];

l = [ones(1,1) zeros(1,15)];
ditheringGrids(:,:,2) = reshape(l(randperm(numel(l))),4,4);

l = [ones(1,2) zeros(1,14)];
ditheringGrids(:,:,3) = reshape(l(randperm(numel(l))),4,4);

l = [ones(1,4) zeros(1,12)];
ditheringGrids(:,:,4) = reshape(l(randperm(numel(l))),4,4);

l = [ones(1,8) zeros(1,8)];
ditheringGrids(:,:,5) = reshape(l(randperm(numel(l))),4,4);

ditheringGrids(:,:,6) = [ 1 1 1 1  ; ...
                          1 1 1 1  ; ...
                          1 1 1 1  ; ...
                          1 1 1 1 ];

end