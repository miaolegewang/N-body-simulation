function produceVideo_andormeda(Filename,num_frames)
   v = VideoWriter(Filename);
   v.FrameRate = 20;
   open(v);
   for i = 1:num_frames
       file = strcat('andromeda/time_',int2str(i),'.png');
       frame = imread(file);
       writeVideo(v,frame);
   end
   close(v);