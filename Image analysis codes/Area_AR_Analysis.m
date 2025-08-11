% radial_ellipse_profile_with_units.m
% -------------------------------------------------------------------------
% 1) Loads updated_mask.npy (labeled image)
% 2) Lets you pick the center by clicking
% 2b) Fits ellipses and overlays them on the mask for visual validation
% 3) Computes ellipse parameters and area
% 4) Bins ellipse areas and aspect ratios by distance from center (pixels)
% 5) Converts distances to microns and areas to µm^2
% 6) Scales error bars by a user-defined factor
% 7) Plots mean ellipse area vs. radius (µm)
% 8) Plots mean aspect ratio vs. radius (µm)
% 9) Exports binned statistics including pixel and µm units
%
% Requires: npy-matlab (readNPY) on your MATLAB path
% -------------------------------------------------------------------------

clear; clc; close all;

%% 1) Load and prepare mask
mask = readNPY('updated_mask.npy');  % must be in current folder
mask = uint16(mask);                 % convert from int64 to supported type

%% 2) Compute ellipse parameters
props = regionprops(mask, ...
    'Centroid', ...            % [x,y]
    'MajorAxisLength', ...     % length of major axis
    'MinorAxisLength', ...     % length of minor axis
    'Orientation');            % ellipse orientation (degrees)

% Extract parameters
centroids        = reshape([props.Centroid],2,[])';        % Nx2 matrix
semiMajor        = [props.MajorAxisLength]'/2;             % Nx1 vector
semiMinor        = [props.MinorAxisLength]'/2;             % Nx1 vector
orientations_deg = [props.Orientation]';                    % Nx1 vector
ellipseAreas_pix = pi * (semiMajor .* semiMinor);           % Nx1, pixels^2
aspectRatios     = ([props.MajorAxisLength]' ./ [props.MinorAxisLength]');  % Nx1

%% 2b) Overlay fitted ellipses on mask (corrected orientation)
% figure('Name','Mask with Fitted Ellipses','Color','w');
% imshow(label2rgb(mask,'jet','k'));
% hold on;
% t = linspace(0,2*pi,100);
% for k = 1:length(props)
%     x0 = centroids(k,1);
%     y0 = centroids(k,2);
%     a  = semiMajor(k);
%     b  = semiMinor(k);
%     % Invert orientation sign to match shape alignment
%     phi = -deg2rad(orientations_deg(k));
%     xe = x0 + a*cos(t)*cos(phi) - b*sin(t)*sin(phi);
%     ye = y0 + a*cos(t)*sin(phi) + b*sin(t)*cos(phi);
%     plot(xe, ye, 'r', 'LineWidth', 1);
% end
% hold off;
% title('Fitted ellipses on mask (orientation corrected)');

%% 3) Display mask and pick center
figure('Name','Select Center','Color','k');
imshow(label2rgb(mask,'jet','k'));
title('Click once to select spheroid center','Color','w');
[xc, yc] = ginput(1);     % single click to choose center
xc = xc(1); yc = yc(1);
hold on;
plot(xc, yc, 'r+', 'MarkerSize', 15, 'LineWidth', 2);
hold off;

%% 4) Compute radial distances (pixels)
dx = centroids(:,1) - xc;
dy = centroids(:,2) - yc;
r  = sqrt(dx.^2 + dy.^2);

%% 5) Define radial bins (pixels)
numBins      = 10;  % adjust as needed
edges        = linspace(0, max(r), numBins+1);
binCents_pix = (edges(1:end-1) + edges(2:end)) / 2;

meanEllipseArea_pix = nan(numBins,1);
semEllipseArea_pix  = nan(numBins,1);
meanAR              = nan(numBins,1);
semAR               = nan(numBins,1);

for i = 1:numBins
    inBin = r >= edges(i) & r < edges(i+1);
    % Ellipse area statistics
    areas_bin = ellipseAreas_pix(inBin);
    nA = numel(areas_bin);
    if nA > 0
        meanEllipseArea_pix(i) = mean(areas_bin);
        semEllipseArea_pix(i)  = std(areas_bin)/sqrt(nA);
    end
    % Aspect ratio statistics
    ar_bin = aspectRatios(inBin);
    nR = numel(ar_bin);
    if nR > 0
        meanAR(i) = mean(ar_bin);
        semAR(i)  = std(ar_bin)/sqrt(nR);
    end
end


% original min/max
orig_min = min(meanAR);
orig_max = max(meanAR);


%% 6) Convert to physical units 
real_radius_um = 208;                % known spheroid radius in microns from ImageJ
max_r_pixel    = max(r);
pixel_to_um    = real_radius_um / max_r_pixel;

binCents_um            = binCents_pix * pixel_to_um;
meanEllipseArea_um2    = meanEllipseArea_pix * pixel_to_um^2;
semEllipseArea_um2     = semEllipseArea_pix  * pixel_to_um^2;


%% 7) Plot mean ellipse area vs. radius (µm) with error bars
figure('Name','Radial Ellipse Area Profile (µm)','Color','w');
errorbar(binCents_um, meanEllipseArea_um2, semEllipseArea_um2, 'o-', ...
    'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('Distance from center (\mum)');
ylabel('Mean ellipse area (\mum^2)');
title('Mean ellipse area vs. radial position');
grid on;

%% 8) Plot mean aspect ratio vs. radius (µm) with scaled error bars
figure('Name','Radial Ellipse Aspect Ratio Profile (µm)','Color','w');
errorbar(binCents_um, meanAR, sem_AR, 's-', ...
    'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('Distance from center (\mum)');
ylabel('Mean ellipse aspect ratio (Major/Minor)');
title('Mean ellipse aspect ratio vs. radial position');
grid on;
