function saveFrames_andromeda(Filename,n1,n2)
  Data = csvread(Filename,0,0);
  index = Data(:,1);
  xdata = Data(:,2);
  ydata = Data(:,3);
  zdata = Data(:,4);
  num_points = n1+n2;
  disp(num_points);
  disp(length(index));
  num_frames = length(index)/num_points;
  disp(num_frames);
  parfor i = 1:num_frames
      index1 = (((i-1)*num_points+1):i*num_points);  %the index of the data in the whole array
      indicator = index(index1);
      x = xdata(index1);
      y = ydata(index1);
      z = zdata(index1);
      A = [indicator, x, y, z];
      A = sortrows(A,1);  % sort the order of A by the index of particles
      x1 = A(1:n1, 2);
      y1 = A(1:n1, 3);
      z1 = -A(1:n1, 4);
      x2 = A((n1+1):num_points, 2);
      y2 = A((n1+1):num_points, 3);
      z2 = -A((n1+1):num_points, 4);
      figure('Color','black');
      plot3(x1,y1,z1,'.','Color','blue','Markersize',0.3);
      hold on;
      plot3(x2,y2,z2,'.','Color','red','Markersize',0.3);
      axis equal;
      %axis([-3,3,-3,3,-3,3]);
      axis off;
      set(gcf,'inverthardcopy','off');
      view(120,0); %rotate the viewpoint
      saveas(gcf,strcat('andromeda/time_',int2str(i)),'png');
      hold off;
  end
      
   