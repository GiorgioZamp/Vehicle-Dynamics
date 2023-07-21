function colorSet = generateColorSetShades(numPlots, initialColor)
% GENERATECOLORSET generates a color set with various darker shades starting from the initial color.
%   numPlots: Number of colors to generate.
%   initialColor (optional): RGB values for the initial color (default: Blue = [0, 0, 1]).
% by: zampieri giorgio

    % Set default initial color if not provided
    if nargin < 2
        initialColor = [0, 0, 1];  % Default: Blue
    end

    % Create a color set of various shades
    colorSet = zeros(numPlots, 3);
    colorSet(1, :) = initialColor;

    % Generate shades
    for i = 2:numPlots
        shade = (i-1)/(numPlots);  % Vary the shade from 0 to 1
        colorSet(i, :) = initialColor * (1 - shade);
    end

end
