%% Programa principal
% Restauracion de imagenes con convolucion

close all
img = im2double(imread('sailsdam.png'));
mask = im2double(imread('sailsmask.png'));

kernel = [0 1 0; 1 0 1; 0 1 0];
kernel_delta = [0 0 0; 0 1 0; 0 0 0]; % kernel delta
kernel_diam = kernel/4; % kernel diamante
kerne_diag = [0.38 0.04 0.04; 0.04 0 0.04; 0.04 0.04 0.38]; %kernel diagonal
a = 0.073235; b = 0.176765;
kernel_diff1 = [a b a; b 0 b; a b a]; % kernel difusion 1 (uso general)
kernel_diff2 = [1 1 1; 1 0 1; 1 1 1]/8; % kernel difusion 2 (uso general)

% Kernels de 7x7
% kernel_gauss7 = fspecial('gaussian',7,1);; % kernel gaussiano 7x7 (para el poste)
% kernel_diff3 = ones(7)/48; kernel4(4,4)=0;% kernel difusion 7x7 (para el poste)

tic
restore=inpainting(img,mask,kernel_diff1,1);
for jj=1:10
restore=inpainting(restore,mask,kernel_diff2,1);
restore=inpainting(restore,mask,kernel_diff1,1);
end
toc
figure(1)
subplot(121)
imshow(img)
subplot(122)
imshow(restore)
pause

tic
restore=inpainting(img,mask,kernel_diff1,0);
for jj=1:10
restore=inpainting(restore,mask,kernel_diff2,0);
restore=inpainting(restore,mask,kernel_diff1,0);
end
toc

figure(2)
subplot(121)
imshow(img)
subplot(122)
imshow(restore)


%% Funciones adicionales
function restore=inpainting(img,mask,kernel,method)
[m,n,z] = size(img); % Dimensiones de la imagen
[x,y] = size(kernel); % Dimensiones del kernel
imgiter = img;

if method == 1

    K = ftkernel(kernel,img); % Se obtiene la fft del kernel

    for ii = 1:m+n

        % Convolución
        imgconv = real(fconv(imgiter,K)); 

        % La convolución se guarda sobre la máscara
        inpaint = imgconv.*(ones(m,n)-mask);

        % Se elimina la máscara de la imagen anterior
        blanks = img.*mask;

        % Se agrega la convolución a la imagen 
        restore = blanks+inpaint;

        % Revisa si la imagen no mejora
        if abs(imgiter-restore)<0.001
            break
        end

        % Guarda la imagen nueva
        imgiter = restore;

        % Visualización
        figure(2)
        imshow(imgiter)
        drawnow
    end

elseif method == 0 
     
     imgconv = zeros(m,n);
     
     for ii = 1:m+n

        % Convolución
        for jj = 1:3
        imgconv(:,:,jj) = conv2(imgiter(:,:,jj),kernel,'same');
        end

        % La convolución se guarda sobre la máscara
        inpaint = imgconv.*(ones(m,n)-mask);

        % Se elimina la máscara de la imagen anterior
        blanks = img.*mask;

        % Se agrega la convolución a la imagen 
        restore = blanks+inpaint;

        % Revisa si la imagen no mejora
        if abs(imgiter-restore)<0.001
            break
        end

        %Guarda la imagen nueva
        imgiter = restore;

        % Visualización
        figure(2)
        imshow(imgiter)
        drawnow

     end
end
end

function imgconv = fconv(img,K)

F = fft2(img); % fft de la imagen

C = F.*K; % Transformada de fourier de la convolución.

imgconv = ifft2(C); % Convolución de imagen y kernel

end

function K=ftkernel(kernel,img)
[x,y] = size(kernel); % Dimensiones del kernel
[m,n,z] = size(img); % Dimensiones de la imagen
center = fix([m/2 n/2]); % Centro de la imagen
kcenter = fix([x/2 y/2]); % Centro del kernel

% El kernel se rellena de ceros para que la ftt tenga las misma dimensiones
% de la fft de la imagen. Dado el algoritmo fft, el primer elemento debe
% corresponder a la frecuencia cero, por lo que se requiere un zero-padding
% especial.

padleft = center(2)-kcenter(2); % Zero-padding a la izquierda
padright = center(2)-(y-kcenter(2)); % Zero-padding a la derecha
padtop = center(1)-kcenter(1); % Zero-padding arriba
padbot = center(1)-(x-kcenter(1)); % Zero-padding abajo
kernel = [zeros(padtop,n);...
    zeros(x,padleft) kernel zeros(x,padright);...
    zeros(padbot,n)]; % Redefinición del kernel relleno de ceros

% En este punto, la frecuencia 0 está en el centro. Esta se puede mover
% usando ifftshift.

K = fft2(ifftshift(kernel)); % fft del kernel

end
