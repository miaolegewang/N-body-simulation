G = 0.287915013;
M1 = 17;    %Mass of Milkway. You can modify it
M2 = 36;    %Mass of Andromeda. You can modify it  
distance = 780;

radius1 = [-369.8+8.5, 615.4 -304.8]/25;  %Ralative position of andromeda. You can modify it. 
                                          %But please comment where the
                                          %data comes from
radius2 = [0,0,0]/25;                     %When position of the Milkway is (0,0,0). Don't modify it.                                        
center = (radius1)/(M1 + M2);               
c_1 = -M2 * center;
c_2 = (-M1/M2)*c_1;

normp = sqrt((369.8-8.5)^2 + (615.4)^2 + (304.8)^2);


n1 = (369.8-8.5)/normp;
n2 = -615.4/normp;
n3 = 304.8/normp;

translate = 0.00408804856;  %unit km/s to 25 kpc / 10^8 year. Don't modify it

vr_mag = 117;     %radius velocity. Feel free to modify it !!!
vt_mag = 40;      %tranverse velocity. Feel free to modify it !!!
vr = vr_mag * [n1, n2, n3];

v1 = vr;
v2 = [1, 0, 0];

v1 = (v1*v2'/(norm(v1)*norm(v1)))*v1;
v2 = v2 - v1;

vt = vt_mag*(v2/norm(v2));

vsum = vr + vt;

vth = sqrt(0.287915013 * 39 / 780/25);


vr_kpcy =  translate * vr;
vt_kpcy =  translate * vt;
vsum_kpcy =  vr_kpcy + vt_kpcy;

time = 2 * pi * (780/25)^2 /sqrt(0.28*78)/20000;

% v_aph = roots([0.25 -G*M/(norm(vt_kpcy)*distance) -0.25*(norm(vsum_kpcy))^2+G*M/distance]);
% r_aph = v_aph(2,1) * distance/norm(vt_kpcy);

v_absolute = vsum_kpcy/2;

centerv = v_absolute/(M1 + M2);
v_1 = -M2 * centerv;
v_2 = (-M1/M2)*v_1;

p_milkway = c_1;    %position of milkway(x,y,z)
v_milkway = v_1;    %velocity of milkway(vx,vy,vx)
p_andromeda = c_2;  %position of milkway(x,y,z)
v_andromeda = v_2;  %velocity of milkway(vx,vy,vz)
