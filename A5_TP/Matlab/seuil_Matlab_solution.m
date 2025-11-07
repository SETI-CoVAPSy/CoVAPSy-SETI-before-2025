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
display('VERSION 1');
tic;  

ima_out_v1=ima;
%A vous de coder ! 
for j=1:taille(2)
for i=1:taille(1)


   
        nr(i,j)=ima(i,j,1)/sqrt(ima(i,j,1)*ima(i,j,1)+ima(i,j,2)*ima(i,j,2)+ima(i,j,3)*ima(i,j,3));
        
        if (nr(i,j)>0.7)
            ima_out_v1(i,j,2)=ima(i,j,1);
        end
    end
end
        

toc;


figure('name','ratio rouge','numbertitle','off');imagesc(nr);colormap gray  %Niveau de rouge

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

display('VERSION 2');
tic;  


ima_out_v2=ima;

nr=ima_r./sqrt(ima_r.^2+ima_g.^2+ima_b.^2);
%A vous de coder !

    for j=1:taille(2)
        for i=1:taille(1)
        if (nr(i,j)>0.7)
            ima_out_v2(i,j,2)=ima(i,j,1);
        end
    end
end
        

toc;


display('VERSION 3');
tic;  
ima_out_v3=ima;

nr=ima_r./sqrt(ima_r.^2+ima_g.^2+ima_b.^2);
image_test=nr>0.7;

image_tmp=ima_g;
image_tmp(image_test)=ima_r(image_test);
ima_out_v3(:,:,2)=image_tmp();
toc;
figure('name','test','numbertitle','off');imagesc(image_test);colormap gray  %Niveau de rouge

figure('name','RGB out version 1','numbertitle','off');image(ima_out_v1);
figure('name','RGB out version 2','numbertitle','off');image(ima_out_v2);
figure('name','RGB out version 3','numbertitle','off');image(ima_out_v3);

%Sauvegarde d'une image au format jpg
imwrite(ima_out_v1,'../Image/ferrari_out_v1.jpg','jpg');
imwrite(ima_out_v2,'../Image/ferrari_out_v2.jpg','jpg');
imwrite(ima_out_v3,'../Image/ferrari_out_v3.jpg','jpg');


%Sauvegarde d'une image au format raw
fid = fopen('../Image/ferrari_out.raw', 'w');
fwrite(fid, ima_out_v3, 'single');
fclose(fid);

