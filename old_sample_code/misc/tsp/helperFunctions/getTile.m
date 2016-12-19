function [tile] = getTile(r,c,I,tileDim) 
    tile = I(1+(r-1)*tileDim:r*tileDim,1+(c-1)*tileDim:c*tileDim,:);
end
