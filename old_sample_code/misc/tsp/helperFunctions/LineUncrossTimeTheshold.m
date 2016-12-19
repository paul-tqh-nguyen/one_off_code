function [output] = LineUncrossTimeTheshold(input, timeLimitInMinutes)

% Uncrosses all lines in our input graph 
% Does not deal with lines on top of each other
% input is of format [X;Y] where the start and end points are the same

%fprintf('\nUncrossing Lines\n');

%tic;

output = input;

numUncrossings = 0;

timeOfLastUncrossing = toc;
minTimeToFindUncrossing = Inf;
maxTimeToFindUncrossing = -Inf;
numFullIterations = 0;

uncrossingHasTakenPlace = true;
while (uncrossingHasTakenPlace) % Keep going through every possible pairing until there are no more crosses to make
    uncrossingHasTakenPlace = false;
    numFullIterations = numFullIterations + 1;
    for i = randperm(size(output,2)-1) % Randomly choosing points in hopes that this makes our search faster
        for j = randperm(size(output,2)-1)
            %fprintf('.');
            if (areCrossed(i,j, output))
                output = uncross(i,j, output);
                numUncrossings = numUncrossings + 1;
                minTimeToFindUncrossing = min(toc - timeOfLastUncrossing, minTimeToFindUncrossing);
                maxTimeToFindUncrossing = max(toc - timeOfLastUncrossing, maxTimeToFindUncrossing);
                uncrossingHasTakenPlace = true;
                %fprintf('.');
                if (mod(numUncrossings,25) == 0)
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

%{
timePassed = toc;
fprintf('\nLine uncrossing finished in %f seconds. There were %d uncrossings. \n', timePassed, numUncrossings);
fprintf('Number of full iterations through all possible pairings was %f. \n', numFullIterations);
fprintf('About %f uncrossings per second. \n', numUncrossings/timePassed);
fprintf('Minimum time to find an uncrossing was %f. \n', minTimeToFindUncrossing);
fprintf('Maximum time to find an uncrossing was %f.\n', maxTimeToFindUncrossing);
%}
end

function [output] = uncross(a_,x_, input)

a = min(a_,x_);
x = max(a_,x_);

b = a+1;
y = x+1;
output = [input(:,1:a) fliplr(input(:,b:x)) input(:,y:end)];

end

function [output] = areCrossed(a,x, input)

if (abs(a-x)<2)
    output = false;
else
    b = a+1;
    y = x+1;

    A = input(:,a);
    B = input(:,b);
    X = input(:,x);
    Y = input(:,y);

    AB = B-A;
    XY = Y-X;

    BX = X-B;
    BY = Y-B;

    YA = A-Y;
    YB = B-Y;

    ABOnDifferentSidesOfXY = CrossProduct2D(XY,YA)*CrossProduct2D(XY,YB) < 0;
    XYOnDifferentSidesOfAB = CrossProduct2D(AB,BX)*CrossProduct2D(AB,BY) < 0;

    output = ABOnDifferentSidesOfXY && XYOnDifferentSidesOfAB;
end

end

function [crossProduct] = CrossProduct2D(A,B)

crossProduct = A(1)*B(2)-A(2)*B(1);

end