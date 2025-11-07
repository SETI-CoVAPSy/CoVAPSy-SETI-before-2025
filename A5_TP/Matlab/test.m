close all;
%Ouverture d'une image au format couleur
ima=single(imread('../Image/ferrari.jpg'));
ima=ima./255;


%Taille d'une image
taille=size(ima);
display(taille);

ima_r=ima(:,:,1);
ima_g=ima(:,:,2);
ima_b=ima(:,:,3);

figure('name','in','numbertitle','off');
ax1=subplot(2,2,1);
ax2=subplot(2,2,2);
ax3=subplot(2,2,3);
ax4=subplot(2,2,4);

%Affichage d'une image couleur avec image
image(ax1,ima),title('RGB');hold on;
%Affichage d'un niveau de couleur de l'image 
imagesc(ax2,ima_r),title('R');colormap gray;hold on %Niveau de rouge
imagesc(ax3,ima_g),title('G');colormap gray;hold on  %Niveau de vert
imagesc(ax4,ima_b),title('B');colormap gray;hold off  %Niveau de bleu

