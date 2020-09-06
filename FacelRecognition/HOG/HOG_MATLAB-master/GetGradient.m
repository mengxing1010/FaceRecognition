function [grad_mag, grad_angle] = GetGradient(im_dx, im_dy)
%Input: im_dx and im_dy are the x and y differential images (size: m × n).
%Output: grad_mag and grad_angle are the magnitude and orientation of the gradient
%images (size: m × n). Note that the range of the angle should be [0, ?), i.e., unsigned
%angle (? == ? + ?).
%Description: Given the differential images, you will compute the magnitude and angle
%of the gradient. Using the gradients, you can visualize and have some sense with the
%image, i.e., the magnitude of the gradient is proportional to the contrast (edge) of 
%the local patch and the orientation is perpendicular to the edge direction as shown in Figure 3.
grad_mag = abs(sqrt(im_dx.^2+im_dy.^2));
grad_angle = atan2(im_dy,im_dx);