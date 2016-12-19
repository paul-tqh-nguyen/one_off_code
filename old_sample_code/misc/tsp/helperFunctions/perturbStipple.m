function [stipple_perturbed] = perturbStipple(stipple, tileDim)

stipple_perturbed = stipple;

stipple_perturbed_height = size(stipple,1);
stipple_perturbed_width = size(stipple,2);

numRowsOfTiles = stipple_perturbed_height/tileDim; % number of rows of tiles
numColumnsOfTiles = stipple_perturbed_width/tileDim; % number of columns of tiles

for row = 1:numRowsOfTiles
    for column = 1: numColumnsOfTiles
        stipple_perturbed(1+(row-1)*tileDim:row*tileDim,1+(column-1)*tileDim:column*tileDim) = matrixShuffle( stipple_perturbed(1+(row-1)*tileDim:row*tileDim,1+(column-1)*tileDim:column*tileDim) );
    end
end

end