function S = GenStroke(im, ks, dirNum)
% ==============================================
%   Compute the Stroke Structure 'S'
%   S = GenStroke(im, ks, dirNum) caculates the direction angle of pixels in  
%  "im", with kernel size "ks", and number of directions "dirNum".
%  
%   Paras:
%   @im        : input image ranging value from 0 to 1.
%   @ks        : kernel size.
%   @dirNum    : number of directions.
%
%           
%   Example
%   ==========
%   im = im2double(imread('npar12_pencil2.bmp'));
%   Iruv = rgb2ycbcr(im);
%   S= GenStroke(Iruv(:,:,1),ks,dirNum);
%   figure, imshow(S)
%
%
%   ==========
%   The code is created based on the method described in
%   "Combining Sketch and Tone for Pencil Drawing Production" Cewu Lu, Li Xu, Jiaya Jia 
%   International Symposium on Non-Photorealistic Animation and Rendering (NPAR 2012), June, 2012

%  image gradients
    [H, W, sc] = size(im);
    if sc == 3
        im = rgb2gray(im);
    end
    imX = [abs(im(:,1:(end-1)) - im(:,2:end)),zeros(H,1)];
    imY = [abs(im(1:(end-1),:) - im(2:end,:));zeros(1,W)];  
    imEdge = imX + imY;

% convolution kernel with horizontal direction 
    kerRef = zeros(ks*2+1);
    kerRef(ks+1,:) = 1;

% classification 
    response = zeros(H,W,dirNum);
    for n = 1 : dirNum
        ker = imrotate(kerRef, (n-1)*180/dirNum, 'bilinear', 'crop');
        response(:,:,n) = conv2(imEdge, ker, 'same');
    end

    [~ , index] = max(response,[], 3); 

 % create the sketch
    C = zeros(H, W, dirNum);
    for n=1:dirNum
        C(:,:,n) = imEdge .* (index == n);
    end

    Spn = zeros(H, W, dirNum);
    for n=1:dirNum
        ker = imrotate(kerRef, (n-1)*180/dirNum, 'bilinear', 'crop');
        Spn(:,:,n) = conv2(C(:,:,n), ker, 'same');
    end

    Sp = sum(Spn, 3);
    Sp = (Sp - min(Sp(:))) / (max(Sp(:)) - min(Sp(:)));
    S = 1 - Sp;