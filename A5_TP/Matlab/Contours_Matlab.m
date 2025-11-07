
%Ouverture d'une image au format raw
fid = fopen('../Image/carre.raw', 'rb');
image_in=fread(fid, 512*512, 'single');
image_in=reshape(image_in,512,512);
fclose(fid);


%Affichage d'une image couleur avec image
figure('name','Image in','numbertitle','off');imagesc(image_in);colormap(gray);

%Taille d'une image
taille=size(image_in);
display(taille);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%tic toc pour mesurer le temps de calcul  
tic;  
hx=[-1 0 1];
hy=[1;0;-1];

h=conv2(hx,hy)
h2D=[0 1 0;-1 0 1;0 -1 0];

image_out=conv2(image_in,h2D,'same');
image_out=abs(image_out);

%pour la convolution regardez l'aide pour conv2

%pour le filtre median regardez l'aide de medfilt2


toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%Affichage d'une image avec imagesc
figure('name','Image out','numbertitle','off');imagesc(image_out);colormap gray;



%Sauvegarde d'une image au format jpg
imwrite(image_out,'../Image/carre_out.jpg','jpg');


%Sauvegarde d'une image au format raw
fid = fopen('../Image/carre_out.raw', 'w');
fwrite(fid, image_out, 'single');
fclose(fid);