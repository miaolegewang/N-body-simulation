function tracenter(Filename,ind_center)
  Data = csvread(Filename,0,0);
  index = Data(:,1);
  xdata = Data(:,2);
  ydata = Data(:,3);
  zdata = Data(:,4);
  tdata = Data(:,8);
  %num_particles = max(index) + 1;
  %mstep = length(xdata)/num_particles;
  position_1 = find(index == 0);
  position_2 = find(index == ind_center);
  xcenter1 = xdata(position_1);
  ycenter1 = ydata(position_1);
  zcenter1 = zdata(position_1);
  xcenter2 = xdata(position_2);
  ycenter2 = ydata(position_2);
  zcenter2 = zdata(position_2);
  tnow = tdata(position_1);
  dist = (xcenter1-xcenter2).^2 + (ycenter1-ycenter2).^2 + (zcenter1 - zcenter2).^2;
  dist = sqrt(dist);
  plot(tnow,dist);
  hold on;
  xlabel('time   unit:3.52e+6 year');
  ylabel('distance  unit:4.5 kpc');
end
