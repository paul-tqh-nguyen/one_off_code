function [J] = GridBasedStippling(I, tileDim, scale, sparsityFactor)

% Similar to Boash and Herman's grid-based method of stippling.
% Only difference is we pre-determine the size of the tiles instead of
% determining the size of the tiles according to the number of tiles we
% want in our image.

%{ 

% Test script

I = imread('monalisa.jpeg');

Iheight = size(I,1);
Iwidth = size(I,2);

scaleFactor = 10;
tileDim = 10;
scale = 10;
sparsityFactor = 250;

J = GridBasedStippling(I, tileDim, scale, sparsityFactor);
output = NNTSP(J); 

Jheight = size(J,1);
Jwidth = size(J,2);

dispayDim = max(Jwidth+1,Jheight+1);

center = [floor(Jwidth/2) floor(-Jheight/2)];
XCOORD = 1;
YCOORD = 2;

plot(output(XCOORD,:),output(YCOORD,:));
padding = min(floor(Iheight/20),floor(Iwidth/20))*scale;
axis([center(XCOORD)-(ceil(dispayDim/2)+padding) center(XCOORD)+(ceil(dispayDim/2)+padding) center(YCOORD)-(ceil(dispayDim/2)+padding) center(YCOORD)+(ceil(dispayDim/2)+padding)]);

%}

% We're going to try to cut our image into a bunch of tileDime x tileDim tiles and then
% determine the density of these tiles. We're then going to try an put
% points there randomly according to the density of the tiles. 
I = fitImage(im2double(rgb2gray(I)), tileDim);

% scale = how much bigger we want the output stippling to be
% sparsityFactor = how much we want to reduce our densities.

Iheight = size(I,1);
Iwidth = size(I,2);

numRowsOfTiles = Iheight/tileDim; % number of rows of tiles
numColumnsOfTiles = Iwidth/tileDim; % number of columns of tiles

J = zeros(size(I)*scale); % J is (scale) times bigger than I

for row = 1:numRowsOfTiles
    for column = 1: numColumnsOfTiles
        tile = getTile(row,column,I, tileDim);
        density = sum(tile(:)) / (tileDim*tileDim);
        % We want to reduce the density by some amount (spartsityFactor) because
        % without it, our image will look almost like our original. 
        density=density/sparsityFactor; 
        %time to put points down
        numTilePixels = scale*tileDim * scale*tileDim;
        numDots = floor(density * numTilePixels);
        newTile = zeros(1,numTilePixels);
        newTile(1:numDots) = ones(1,numDots);
        newTile = shuffleMatrix(reshape(newTile, tileDim*scale, tileDim*scale));
        J(1+(row-1)*tileDim*scale:row*tileDim*scale,1+(column-1)*tileDim*scale:column*tileDim*scale,:) = newTile;
    end
end

%{
%figure, imshow(J)
J = imresize(J, 1/scale);
J(J < mean(J(:))) = 0;
J(J ~= 0) = 1;
figure, imshow(J)
%}

end

function [shuffledM] = shuffleMatrix(M)
    shuffledM = reshape(M(randperm(numel(M))),size(M,1),size(M,2));
end