clear
close all
filenames = 'trajdata.csv';
data = readmatrix(filenames);
ind = 1:length(data)/9;
ind = randi(length(data),1,1000);
xf = data(ind,7)*100;
yf = data(ind,8)*100;
tf = data(ind,9);
r = 0.0008*100;
conv = 1;
C = parula(1000);
rows = size(C);
P = randperm(rows(1));
C = C(P,:);
hold on
for i=1:length(ind)
    plot([0 xf(i)], [0 yf(i)],'Color',C(i,:))
    hold on
    R = [cosd(tf(i)), -sind(tf(i)); sind(tf(i)), cosd(tf(i))];
    rect = R*[-r*conv,-r*conv,r*conv,r*conv;-r,r,r,-r] + [xf(i); yf(i)];
    fill(rect(1,:), rect(2,:),C(i,:))
end
xlabel('x (cm)')
ylabel('y (cm)')
set(gca,'FontSize',12)