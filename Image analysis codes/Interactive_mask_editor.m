% interactive_mask_editor.m
% Make sure npy-matlab is in your path
% https://github.com/kwikteam/npy-matlab

clear; clc;

% Load labeled mask
mask = readNPY('mask.npy');  % Ensure mask.npy is in current folder

% Display original mask
figure('Name', 'Interactive Mask Editor');
hImg = imshow(label2rgb(mask, 'jet', 'k'));
title('Click inside a mask to delete it. Right-click or press Enter to finish.');

deleted_labels = [];

% Interactive loop
while true
    [x, y, button] = ginput(1);
    
    if isempty(button) || button ~= 1  % Exit on right-click or Enter
        break;
    end

    x = round(x);
    y = round(y);

    % Bounds check
    if x < 1 || x > size(mask,2) || y < 1 || y > size(mask,1)
        continue;
    end

    label = mask(y, x);

    if label ~= 0
        fprintf('Deleting label: %d\n', label);
        mask(mask == label) = 0;
        deleted_labels(end+1) = label;
        set(hImg, 'CData', label2rgb(mask, 'jet', 'k'));
        drawnow;
    end
end

% Save updated mask
writeNPY(mask, 'updated_mask.npy');
fprintf('Done. Deleted %d labels.\n', numel(deleted_labels));
fprintf('Saved to updated_mask.npy\n');

%% fit_ellipses_from_mask.m
% 1) Load the updated mask
mask = readNPY('updated_mask.npy');    % must be in current folder

% 2) Convert from int64 to a supported type
mask = uint16(mask);  % uint16 handles up to 65k labels

% 3) Compute ellipse properties
props = regionprops(mask, ...
    'Centroid', ...            % [x,y]
    'MajorAxisLength', ...     % length of major axis
    'MinorAxisLength', ...     % length of minor axis
    'Orientation', ...         % angle in degrees
    'Area', ...                % pixel area
    'Eccentricity');           % shape eccentricity

% 4) Visualize: color-label mask + overplot ellipses
figure('Name','Ellipse Fits','Color','k');
imshow(label2rgb(mask,'jet','k')); 
hold on;
title('Fitted Ellipses','Color','w','FontSize',14);

for i = 1:numel(props)
    c = props(i).Centroid; 
    a = props(i).MajorAxisLength/2;
    b = props(i).MinorAxisLength/2;
    ang = -props(i).Orientation;  % negative to match MATLAB coords

    % Parametric ellipse
    theta = linspace(0,2*pi,200);
    xy = [a*cos(theta); b*sin(theta)];
    R = [cosd(ang) -sind(ang); sind(ang) cosd(ang)];
    xy_rot = R * xy;

    % Translate and plot
    plot(xy_rot(1,:) + c(1), xy_rot(2,:) + c(2), 'w-', ...
         'LineWidth', 1.5);
end

hold off;

% 5) Export ellipse parameters
T = struct2table(props);
writetable(T, 'ellipse_fits.csv');
fprintf('Ellipse parameters saved to ellipse_fits.csv\n');
