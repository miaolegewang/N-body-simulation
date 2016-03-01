function saveFrames(Filename)
  Data = csvread(Filename,0,1);
  xdata = Data(:,2);
  ydata = Data(:,3);
  zdata = Data(:,4);
  num_points = max(Data(:,1)) + 1;
  num_frames = length(xdata)/num_points;
  parfor i = 1:num_frames
      index = (((i-1)*num_points+1):i*num_points);
      x = xdata(index);
      y = ydata(index);
      z = zdata(index);
      plot3(x,y,z,'.','Color','blue','Markersize',0.7);
      axis equal;
      axis([-1.5,1.5,-1.5,1.5,-1.5,1.5]);
      %axis equal;
      axis off;
      %set(gcf,'inverthardcopy','off');
      saveas(gcf,strcat('myimage/time_',int2str(i)),'png');
  end
      
   