function colorSet = generateColorSetTrans(numPlots, initialColor, finalColor)
% GENERATECOLORSET generates a color set with various shades starting from the initial color and
% ending to final color interpolating linearly.
%   numPlots: Number of colors to generate.
%   initialColor (optional): RGB values for the initial color (default: Blue = [0, 0, 1]).
%   finalColor (optional)  : RGB values for the final color (default: Green = [0, 1, 0]).
% by: zampieri giorgio

    % Set default colors if not provided
    if nargin < 2
        initialColor = [0, 0, 1];  % Default: Blue
        finalColor   = [0, 1, 0];  % Default: Green
    end
    if nargin < 3
        finalColor   = [0, 1, 0];  % Default: Green
    end

    % Create a color set of various shades
    colorSet = zeros(numPlots, 3);
    colorSet(1, :) = initialColor;

    % Generate shades
    v = finalColor - initialColor; % vector connecting points
    for i = 2:numPlots
        shade = (i-1)/(numPlots);
        colorSet(i, :) = initialColor + shade*v/norm(v);
    end

    % Plotting the series of colors (optional)
    figure;
    hold on;

    for i = 1:numPlots
        x = 1:10;
        y = x * i;
        plot(x, y, 'Color', colorSet(i, :));
    end

    hold off;

end
