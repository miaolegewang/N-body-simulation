function animate(input)
M = csvread(input);
num = max(M(:, 1)) + 1;
%f = figure('Color', 'k');
%axis([-5 5 -5 5 -5 5]);
%axis off;
for i = 1 : num / 2
    h(i) = line(0, 0, 0);
    set(h(i), 'Marker', '.', 'Color', 'b', 'MarkerSize', 1);
end
for i = num / 2 + 1 : num
    h(i) = line(0, 0, 0);
    set(h(i), 'Marker', '.', 'Color', 'r', 'MarkerSize', 1);
end
for i = 1 : num : size(M)
    for j = 1 : num
        set(h(M(i + j - 1, 1) + 1), 'XData', M(i + j - 1, 2), 'YData', M(i + j - 1, 3), 'ZData', M(i + j - 1, 4));
    end
    drawnow
end
end