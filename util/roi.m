% Get the region of interest (ROI) of the image
% return a mask of the image

function mask=roi(img,pcnt,region)
if ~exist('pcnt','var') || isempty(pcnt)
    pcnt=0.75;
end
if ~exist('region','var') || isempty(region)
    region='largest';
end

% covnert RGB to grayscale
imsz=size(img);
if length(imsz)==3
    img=rgb2gray(img);
    imsz=imsz(1:2);
end
% obtain boundaries
bdy=bwboundaries(img>quantile(img(:),pcnt),'noholes');
if contains(region,'largest')
    len=cellfun(@length,bdy); [~,ind]=max(len);
    bdy=bdy(ind);
end
% obtain mask of ROI from boundaries
mask=sparse(imsz(1),imsz(2));
for k=1:length(bdy)
    b_k=bdy{k};
    msk_k=sparse(poly2mask(b_k(:,2),b_k(:,1),imsz(1),imsz(2)));
    mask=mask|msk_k;
end

end