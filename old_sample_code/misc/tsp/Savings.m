function [output] = Savings(input, timeLimitInMinutes)

tic;

output = input;

numSwaps = 0;

timeOfLastUncrossing = toc;
minTimeToFindUncrossing = Inf;
maxTimeToFindUncrossing = -Inf;
numFullIterations = 0;

swappingHasTakenPlace = true;
while (swappingHasTakenPlace) % Keep going through every possible pairing until there are no more crosses to make
    swappingHasTakenPlace = false;
    numFullIterations = numFullIterations + 1;
    for i = randperm(size(output,2)-1) % Randomly choosing points in hopes that this makes our search faster
        for j = randperm(size(output,2)-1)
            %fprintf('.');
            if (swapNecessary(i,j, output))
                output = swap(i,j, output);
                numSwaps = numSwaps + 1;
                minTimeToFindUncrossing = min(toc - timeOfLastUncrossing, minTimeToFindUncrossing);
                maxTimeToFindUncrossing = max(toc - timeOfLastUncrossing, maxTimeToFindUncrossing);
                swappingHasTakenPlace = true;
                %fprintf('.');
                if (mod(numSwaps,25) == 0)
                    %fprintf('\n');
                end
                timeOfLastUncrossing = toc;
            end
            if (toc > timeLimitInMinutes*60)
                break;
            end
        end
        if (toc > timeLimitInMinutes*60)
            break;
        end
    end
    if (toc > timeLimitInMinutes*60)
        break;
    end
    %fprintf('.');
end

end

function [output] = swap(a_,x_, input)

a = min(a_,x_);
x = max(a_,x_);

b = a+1;
y = x+1;
output = [input(:,1:a) fliplr(input(:,b:x)) input(:,y:end)];

end

function [isNecessary] = swapNecessary(a_,x_, input)

a = min(a_,x_);
x = max(a_,x_);

if (abs(a-x)<2)
    isNecessary = false;
else
    b = a+1;
    y = x+1;

    A = input(:,a);
    B = input(:,b);
    X = input(:,x);
    Y = input(:,y);
    
    Dab = sum( (A-B).^2 );
    Dax = sum( (A-X).^2 );
    Dxy = sum( (X-Y).^2 );
    Dby = sum( (B-Y).^2 );
    
    isNecessary = (Dax+Dby < Dab+Dxy);
end

end