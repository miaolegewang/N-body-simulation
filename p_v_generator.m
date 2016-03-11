G = 0.287915013;
M = 19.5;
distance = 780;

radius1 = [-369.8+8.5, 615.4 -304.8];  %distance between milkway and andromeda
radius2 = [-8.5,0,0];
center = (radius1 + radius2)/2;
c_1 = (radius1 - center)/25;
c_2 = (radius2 - center)/25;

normp = sqrt((369.8-8.5)^2 + (615.4)^2 + (304.8)^2);


n1 = (369.8-8.5)/normp;
n2 = -615.4/normp;
n3 = 304.8/normp;

translate = 0.00408804856;

vr_mag = 117;     %radius velocity
vt_mag = 40;      %tranverse velocity
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

p_milkway = c_1;
v_milkway = -v_absolute;
p_andromeda = c_2;
v_andromeda = v_absolute;
