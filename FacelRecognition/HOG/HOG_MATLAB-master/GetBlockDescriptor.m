function ori_histo_normalized = GetBlockDescriptor(ori_histo, block_size)
%Input: ori_histo is the histogram of oriented gradients without normalization. block_size
%is the size of each block (e.g., the number of cells in each row/column), which is a positive integer.
%Output: ori_histo_normalized is the normalized histogram (size: (M?(block_size?
%1)) × (N ? (block_size ? 1)) × (6 × block_size2).
%Description: To account for changes in illumination and contrast, the gradient strengths
%must be locally normalized, which requires grouping the cells together into larger, spatially connected blocks (adjacent cells). Given the histogram of oriented gradients, you
%apply L2 normalization as follow

M = size(ori_histo,1);
N = size(ori_histo,2);
nbins = size(ori_histo,3);
e = 0.001;
ori_histo_normalized = zeros(M-(block_size-1),N-(block_size-1),nbins*block_size*block_size);

for i = 1 : M-block_size
    for j = 1 : N-block_size
        hog_vector = [];
        norm_vector = [];
        for k = 1:i+(block_size-1);
            for l = 1:j+(block_size-1);
                for c = 1:nbins
                    % concatenate each entry in 2x2x6 block to form long vector
                    hog_vector = [hog_vector; ori_histo(k,l,c)];
                    
                    % normalize the vector
                    norm_vector = hog_vector/sqrt(sum(hog_vector.^2)+e.^2);
                    
                end
            end
        end
        % assign the normalized histogram to ori_histo_normalized(1,1)
        ori_histo_normalized(i,j,:) = norm_vector;
        pause
    end
end
end