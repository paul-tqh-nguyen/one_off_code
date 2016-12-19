function [totalTourDistance] = calculateTourDistance(input)

input_reindexed = [input(:,end) input(:,1:end-1)];

totalTourDistance = sum(sum((input-input_reindexed).^2,1).^0.5);

end