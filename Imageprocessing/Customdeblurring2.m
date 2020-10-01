clc
close all;

I1=imread('C:\Users\manan\Desktop\fruits.jpg');
% im2 double converts image to double precision
I = im2double(imread('C:\Users\manan\Desktop\fruits.jpg')); 
figure, imshow(I)
title('Input Image');

% simulate blur 
LEN = 21;
THETA = 11;
%Creating a filter to get blurred image
%Crating a filter wrt to the linear motion of camera where len specifies
%length of motion and thetha specifies angle of motion in degrees
PSF = fspecial('motion', LEN, THETA);
% imfilter filters the multidimensional array I with the multidimensional filter PSF.
% Here imfilter is filtering accoridung to convolution and the input values are circular.
blurred = imfilter(I, PSF, 'conv', 'circular'); 
figure, imshow(blurred);
title('Blurred Image');

% add gaussian noise to the image 
noise_mean = 0;
noise_var = 0.0001;
% im noise adds ' ' noise with a noise density wrt to noise_var on blurred
% image
blurred_noisy = imnoise(blurred, 'gaussian', noise_mean, noise_var); 
figure,imshow(blurred_noisy);
title('Blurred and Noisy (Gaussian) Image')

%Blurred_noisy = Image with gaussian noise

% adding poisson noise to the image blurred_noisy =
imnoise(blurred_noisy,'poisson')
blurred_noisy = imnoise(blurred, 'gaussian', noise_mean, noise_var);
figure, imshow(blurred_noisy)
title('Blurred and Noisy (Gaussian + Poisson) Image')


% customised guassian deblurring 
%im = gray value of blurred image
im=rgb2gray(blurred);
fc=100;
%fft2(X) returns the two-dimensional Fourier transform of a matrix using a fast Fourier transform algorithm,
%fftshift(X) rearranges a Fourier transform X by shifting the zero-frequency component to the center of the array.
imf= fftshift(fft2(im)); 
% co = coloumn
% ro= row
[co,ro]=size(im);
%out is making matrix of size co*ro =0
out = zeros(co,ro); 
cx = round(co/2);
cy = round(ro/2);
H = zeros(co,ro);
 for i = 1 : co
     for j = 1 : ro
         d = (i-cx).^2 + (j-cy).^2;
         H(i,j) = exp(-d/2/fc/fc);
     end
 end
outf= imf.*H;
%ifft= inverse fourier transform
out=abs(ifft2(outf)); 
figure, imshow(out);
title('Blurred and Noisy (Custom Guassian Deblurring) Image')


% deblur

estimated_nsr = noise_var / var(out(:));
J = deconvwnr(blurred_noisy, PSF, estimated_nsr);
figure, imshow(J)
title('Wiener Filter')

% second pass 
V = .002;
PSF = fspecial('gaussian',5,5);
luc3 = deconvlucy(J,PSF,15); 
figure, imshow(luc3)
title('Restored Image using Lucy-Richardson algorithm');

% remove noise
H = fspecial('gaussian',2, 5); 
J = imfilter(J, H);
figure, imshow(J);
title('Removed Gaussian Image')

% harmonic mean 
dim=3;
m = harmmean(J);
harmmean(J,dim); 
figure, imshow(J);
title('Harmonic mean')

stretched_truecolor = imadjust(J,stretchlim(J)); 
figure, imshow(stretched_truecolor);
title('Truecolor Composite after Contrast Stretch')

% sharpen
H = padarray(2,[2 2]) - fspecial('gaussian' ,[5 5],2); 
sharpened =imfilter(stretched_truecolor,H);
figure, imshow(sharpened);
title('Sharpened image')
