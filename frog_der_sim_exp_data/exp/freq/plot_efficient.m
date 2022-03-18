clc
clear all
close all
Dir = 'C:\Users\huang\Downloads\frog_data\Data\efficient\sim\'
Dir1 = 'C:\Users\huang\Downloads\frog_data\Data\efficient\exp\'

files = dir([Dir, '*.txt']); % Locate the files
Files = dir([Dir1, '*.mat']); % Locate the files
Nfiles = length(files);
colpos = [247 148 30;0 166 81; 237 28 36; 0 174 239; 0 250 250; 100 50 150; 150 150 200; 0 0 0; 200 30 150; 50 230 50]/255; % colors
for i = 1:length(files)
    name{i} = files(i).name;
    Name{i} = Files(i).name;
end
figure()
hold on
for k=1:Nfiles
    f = load([Dir, name{k}]);
    t = f(:,1);
    d = f(:,3);
    hold on
    indS = round(5/(k*0.1+0.3)*1000)
    indF = round(8/(k*0.1+0.3)*1000);
    d(indF)
    d(indS)
    t(indF)
    t(indS)
    v(k) = (d(indF) - d(indS)) / (t(indF) - t(indS)) * 1000
    time_sim = t(indS:indF) - t(indS)*ones(indF-indS+1,1);
    dist_sim = (d(indS:indF) - d(indS)*ones(indF-indS+1,1))*1000;
    plot(time_sim, dist_sim, '-', 'LineWidth', 2.5, 'Color', colpos(k+2,:))
    F = load([Dir1, Name{k}]);
    time_exp = F.time;
    dist_exp = F.dist;
    plot(time_exp, dist_exp, '--', 'LineWidth', 2.5, 'Color', colpos(k+2,:))
end
legend('0.4Hz sim', '0.4Hz exp','0.5Hz sim', '0.5Hz exp')
xlabel('Time [s]')
ylabel('Distance [mm]')
% figure(4)
freq = [0.4,0.5];
figure()
X = categorical({'0.4Hz, sim', '0.4Hz, exp', ...
    '0.5Hz, sim','0.5Hz, exp'});
% X = reordercats(X,{'A1, sim', 'A1, exp', ...
%     'A1&2, sim','A1&2, exp', 'A1&3, sim',...
%     'A1&3, exp', 'A1&4, sim', 'A1&4, exp',...
%     'A1&3&5, sim', 'A1&3&5, exp'});
fileName = 'speed.xlsx';
sheet = 2;
num = xlsread(['C:\Users\huang\Downloads\STAR_frog\Data\exp\',fileName],sheet);
v_exp = num(1:2,2);
v = [v(1);v_exp(1);v(2);v_exp(2)];
b = bar(X,v, 0.5)
% b(2).FaceColor = 'red'
ylabel('speed [mm/s]')
ylim([0 40])
% plot(freq40,v, 'o', 'MarkerSize', 8, 'Color', colpos(1,:))
% ylim([0 60])

% sheet = 4;
% num = xlsread(['C:\Users\huang\Downloads\STAR_frog\Data\exp\',fileName],sheet);
% v_exp = num(:,2);
% hold on
% plot(freq40,v_exp, '*', 'MarkerSize', 8, 'Color', colpos(2,:))
% xlabel('Speed [mm/s]')
% ylabel('Frequency [Hz]')
% legend('sim', 'exp')