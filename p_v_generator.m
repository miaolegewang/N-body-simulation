G = 1;
M1 = 38.2352941;    %Mass of Milkway. 5.1*10^10
M2 = 38.2352941;    %Mass of Andromeda. 5.1*10^10
distance = 780; %Relative position of Milkyway and Andromeda
l = 121; %Adromeda position in galactic coordinate(l,b);
b = -23;
vr_mag = 117;     %relative radius velocity. km/s
vt_mag = 40;      %relative tranverse velocity.  km/s
translate = 0.00454545455;  %unit 1 km/s to 4.5 kpc / 2*10^7 year, also 220 km/s. 


x1 = distance * cos((b/360)*2*pi) * cos((l/360)*2*pi)/4.5;  %translate to xyz unit 4.5 kpc
y1 = distance * cos((b/360)*2*pi) * sin((l/360)*2*pi)/4.5;
z1 = distance * sin((b/360)*2*pi)/25;

radius1 = [x1, y1, z1];  
radius2 = [0,0,0];                                                             
center = (radius1)/(M1 + M2);               
c_1 = -M2 * center;
c_2 = (-M1/M2)*c_1;

normp = sqrt((x1)^2 + (y1)^2 + (z1)^2);

n1 = x1/normp;
n2 = y1/normp;
n3 = z1/normp;

vr = vr_mag * [n1, n2, n3];

v1 = vr;
v2 = [1, 0, 0];

v1 = (v1*v2'/(norm(v1)*norm(v1)))*v1;
v2 = v2 - v1;

vt = vt_mag*(v2/norm(v2));

vsum = vr + vt;

vr_kpcy =  translate * vr;
vt_kpcy =  translate * vt;
vsum_kpcy =  vr_kpcy + vt_kpcy;


v_absolute = vsum_kpcy;

centerv = v_absolute/(M1 + M2);
v_1 = -M2 * centerv;
v_2 = (-M1/M2)*v_1;

p_milkway = c_1;    %position of milkway(x,y,z)
v_milkway = v_1;    %velocity of milkway(vx,vy,vx)
p_andromeda = c_2;  %position of milkway(x,y,z)
v_andromeda = v_2;  %velocity of milkway(vx,vy,vz)
