function ori_histo = BuildHistogram(grad_mag, grad_angle, cell_size)
%Input: grad_mag and grad_angle are the magnitude and orientation of the gradient
%images (size: m × n); cell_size is the size of each cell, which is a positive integer.
%Output: ori_histo is a 3D tensor with size M × N × 6 where M and N are the
%number of cells along y and x axes, respectively, i.e., M = m/cell_size and N =
%n/cell_size where b·c is the round-off operation as shown in Figure 4(a).
%Description: Given the magnitude and orientation of the gradients per pixel, you can
%build the histogram of oriented gradients for each cell.
m = size(grad_angle,1);
n = size(grad_angle,2);
M = floor(m/cell_size);
N = floor(n/cell_size);
theta = [15 45 75 105 135 165];
nbins = size(theta,2);
%ori_histo = zeros(M,N,nbins);
ori_histo = zeros(M,N,nbins);
% change the gradient angle value from rad to angle
grad_angle = radtodeg(grad_angle);
% Divide the image into cell_size blocks
for i = 0 : M-1
    for j = 0 : N-1
        u = (i*cell_size)+1;
        v = (j*cell_size)+1;
        %A(u:u+cell_size-1,v:v+cell_size-1)
        for k = u:u+cell_size-1
            for l = v:v+cell_size-1
                % check the angles
                if (grad_angle(k,l) >= theta(1)) && (grad_angle(k,l) < theta(2)) % if the angle is [15,45)
                    ori_histo(i+1,j+1,2) = ori_histo(i+1,j+1,2) + grad_mag(k,l);
                elseif (grad_angle(k,l) >= theta(2)) && (grad_angle(k,l) < theta(3)) % if the angle is [45,75)
                    ori_histo(i+1,j+1,3) = ori_histo(i+1,j+1,3) + grad_mag(k,l);
                elseif (grad_angle(k,l) >= theta(3)) && (grad_angle(k,l) < theta(4)) % if the angle is [75,105)
                    ori_histo(i+1,j+1,4) = ori_histo(i+1,j+1,4) + grad_mag(k,l);
                elseif (grad_angle(k,l) >= theta(4)) && (grad_angle(k,l) < theta(5)) % if the angle is [105,135)
                    ori_histo(i+1,j+1,5) = ori_histo(i+1,j+1,5) + grad_mag(k,l);
                elseif (grad_angle(k,l) >= theta(5)) && (grad_angle(k,l) < theta(6)) % if the angle is [135,165)
                    ori_histo(i+1,j+1,6) = ori_histo(i+1,j+1,6) + grad_mag(k,l);
                else % if the angle is [165,180) U [0,15)
                    ori_histo(i+1,j+1,1) = ori_histo(i+1,j+1,1) + grad_mag(k,l);
                end
            end
        end
    end
    
end
