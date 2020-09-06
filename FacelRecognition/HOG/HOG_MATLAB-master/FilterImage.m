function [im_filtered] = FilterImage(im, filter)
%Input: im is the gray scale m × n image (Figure 2(a)) converted to double (refer to
%im2double built-in function); filter is a filter (k × k matrix)
%Output: im_filtered is m × n filtered image. You may need to pad zeros on the
%boundary on the input image to get the same size filtered image.
%Description: Given an image and filter, you will compute the filtered image. Given the
%two functions above, you can generate differential images by visualizing the magnitude
%of the filter response as shown in Figure 2(b) and 2(c).
im = im2double(im);
im_filtered = zeros(size(im));
center_k = floor(size(filter,1)/2)+1;

center_l = floor(size(filter,2)/2)+1;
for i = 1 : size(im,1)
    for j = 1 : size(im,2)
    % filtering
        v = 0;
        for k = 1 : size(filter,1)
            for l = 1 : size(filter,2)
                i1 = i + k - center_k;
                j1 = j + l - center_l;
                if i1 <= 0 || i1 > size(im_filtered,1) || j1 <= 0 || j1 > size(im_filtered,2)
                    continue;
                end
                v = v + im(i1,j1)*filter(k,l);
            end
        end
        im_filtered(i,j) = v;
    end
end