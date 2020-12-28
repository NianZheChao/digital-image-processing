img = imread('kenan.jpg')
subplot(331);
imshow(img), title('Original Image')

PSF = fspecial('gaussian',15,15);
blur = imfilter(img,PSF,'replicate');
subplot(332);imshow(blur);title('Filter image');

motion_noise = fspecial('disk', 7);

luc1 = deconvlucy(img,motion_noise);
subplot(333); imshow(luc1);
title('Disk and Lucy');

LEN = 9; THETA = 1;
motion_noise2 = fspecial('motion', LEN, THETA);


luc2 = deconvlucy(blur,motion_noise2);
subplot(334); imshow(luc2);
title('Motion and Lucy');