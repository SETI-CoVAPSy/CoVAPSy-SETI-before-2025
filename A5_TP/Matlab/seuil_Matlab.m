close all;

%Ouverture d'une image au format couleur
ima=single(imread('../Image/ferrari.jpg'));
ima=ima./255;

%Affichage d'une image couleur avec image
figure('name','RGB in','numbertitle','off');image(ima);


%Taille d'une image
taille=size(ima);
display(taille);

ima_r=ima(:,:,1);
ima_g=ima(:,:,2);
ima_b=ima(:,:,3);

%Affichage d'un niveau de couleur de l'image 
figure('name','R','numbertitle','off');imagesc(ima_r);colormap gray  %Niveau de rouge
figure('name','G','numbertitle','off');imagesc(ima_g);colormap gray  %Niveau de vert
figure('name','B','numbertitle','off');imagesc(ima_b);colormap gray  %Niveau de bleu

%Taille d'une image
taille=size(ima);
display(taille);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%tic toc pour mesurer le temps de calcul  
tic;  

ima_out=ima;
        

toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



figure('name','RGB out','numbertitle','off');image(ima_out);

%Sauvegarde d'une image au format jpg
imwrite(ima_out,'../Image/ferrari_out.jpg','jpg');



%Sauvegarde d'une image au format raw
fid = fopen('../Image/ferrari_out.raw', 'w');
fwrite(fid, ima_out, 'single');
fclose(fid);

