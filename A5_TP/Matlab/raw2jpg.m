function raw2jpg(name_proc)
%fid = fopen('ferrari.raw', 'r');
file_name_in=['../Image/ferrari_out_' name_proc '.raw'];
fid = fopen(file_name_in, 'r');
ima=single(fread(fid,1280*960*3, 'single'));
fclose(fid);

ima=reshape(ima,960,1280,3);
ima=ima./255;
figure('name',name_proc,'numbertitle','off');image(ima);

file_name_out=['../Image/ferrari_out_' name_proc '.jpg'];
imwrite(ima,file_name_out,'jpg');
end