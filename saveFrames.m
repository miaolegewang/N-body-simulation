function saveFrames(Filename)
  Data = csvread(Filename,0,0);
  index = Data(:,1);
  xdata = Data(:,2);
  ydata = Data(:,3);
  zdata = Data(:,4);
  num_points = max(Data(:,1)) + 1;
  num_frames = length(xdata)/num_points;
  disp(num_frames);
  num_particles = num_points/2;
  parfor j = 1:num_frames
      galaxy_A = zeros(num_particles,3);
      galaxy_B = zeros(num_particles,3);
      for nrow = 1:num_points
          i = nrow + (j-1) * num_points;
          currindex = index(i) + 1;
          if (currindex <= num_particles)
              galaxy_A(currindex,:) = [xdata(i) ydata(i) zdata(i)];
          else
              galaxy_B(currindex-num_particles,:) = [xdata(i) ydata(i) zdata(i)];
          end
      end
      figure('Color','black');
      plot3(galaxy_A(:,1),galaxy_A(:,2),-galaxy_A(:,3),'.','Color',[0.2 0.6 1.0],'Markersize',0.7);
      hold on;
      plot3(galaxy_B(:,1),galaxy_B(:,2),-galaxy_B(:,3),'.','Color','red','Markersize',0.7);
      axis equal;
      axis([-200,200,-200,200,-200,200]);
      axis off;
      set(gcf,'inverthardcopy','off');
      view(0,120); %rotate the viewpoint
      saveas(gcf,strcat('andromeda_3b/time_',int2str(j)),'png');
      hold off;
      galaxy_A = [];
      galaxy_B = [];
      clf;
  end
end