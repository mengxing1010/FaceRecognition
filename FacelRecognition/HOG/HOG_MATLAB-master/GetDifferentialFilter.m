function [filter_x, filter_y] = GetDifferentialFilter(kernel)
%Input: none.
%Output: filter_x and filter_y are 3×3 filters that differentiate along x and y directions, respectively.
%Description: You will compute the gradient by differentiating the image along x and
%y directions. This code will output the differential filters.
syms x y

%Sobel filter
%filter_x = [1 0 -1; 2 0 -2; 1 0 -1];
%filter_y = [1 2 1; 0 0 0; -1 -2 -1];

%Scharr filter
%filter_x = [3 0 -3; 10 0 -10; 3 0 -3];
%filter_y = [3 10 3; 0 0 0; -3 -10 -3];

%Prewitt
%filter_x = [1 0 -1; 1 0 -1; 1 0 -1];
%filter_y = [1 1 1; 0 0 0; -1 -1 -1];

%Roberts
filter_x = [0 1; -1 0];
filter_y = [1 0; 0 -1];
end

%Laplacian
%filter = [0 -1 0; -1 4 -1; 0 -1 0];
%filter = [-1 -1 -1; -1 8 -1; -1 -1 -1];