function [J] = fitImage(I, tileDim)

% Let's make sure our image we can fit in a bunch of tileDimxtileDim cells
% by cutting down the image. We're going to assume the important stuff is
% in the middle of the image, so we'll just cut down the useless edge
% columns and rows

I = im2double(I);

Iheight = size(I,1);
Iwidth = size(I,2);

rowsToCut = mod(Iheight,tileDim);

if (mod(rowsToCut, 2) == 1) % if we have an odd number of rows to cut
    I = I(1:end-1,:,:); % Cut the bottom row
    rowsToCut = rowsToCut - 1;
end

% Now the number of rows we have to cut is even
if (rowsToCut ~= 0)
    I = I(rowsToCut/2+1:end-rowsToCut/2,:,:);
end

columnsToCut = mod(Iwidth,tileDim);

if (mod(columnsToCut, 2) == 1)
    I = I(:,1:end-1,:);
    columnsToCut = columnsToCut - 1;
end

if (columnsToCut ~= 0)
    I = I(:,columnsToCut/2+1:end-columnsToCut/2,:);
end

J = I;

end