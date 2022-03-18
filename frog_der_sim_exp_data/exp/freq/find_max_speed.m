clc
clear all
close all

Dir = 'C:\Users\huang\Downloads\STAR_frog\Data\sim\freq\40\'

files = dir([Dir, '*.txt']); % Locate the files
Nfiles = length(files);
colpos = [247 148 30;0 166 81; 237 28 36; 0 174 239; 0 250 250; 100 50 150; 150 150 200; 0 0 0; 200 30 150; 50 230 50]/255; % colors
for i = 1:length(files)
    name{i} = files(i).name;
end
name = natsortfiles(name);

for k=1:Nfiles
    f = load([Dir, name{k}]);
    t = f(:,1);
    d = f(:,3);
    hold on
    indS = round(5/(k*0.1+1.4)*1000);
    indF = round(8/(k*0.1+1.4)*1000);
    v40(k) = (d(indF) - d(indS)) / (t(indF) - t(indS)) * 1000;
end

Dir = 'C:\Users\huang\Downloads\STAR_frog\Data\sim\freq\50\'

files = dir([Dir, '*.txt']); % Locate the files
Nfiles = length(files);
colpos = [247 148 30;0 166 81; 237 28 36; 0 174 239; 0 250 250; 100 50 150; 150 150 200; 0 0 0; 200 30 150; 50 230 50]/255; % colors
for i = 1:length(files)
    name{i} = files(i).name;
end
name = natsortfiles(name);

for k=1:Nfiles
    f = load([Dir, name{k}]);
    t = f(:,1);
    d = f(:,3);
    hold on
    indS = round(5/(k*0.1+1.6)*1000);
    indF = round(8/(k*0.1+1.6)*1000);
    v50(k) = (d(indF) - d(indS)) / (t(indF) - t(indS)) * 1000;
end

Dir = 'C:\Users\huang\Downloads\STAR_frog\Data\sim\freq\30\'

files = dir([Dir, '*.txt']); % Locate the files
Nfiles = length(files);
colpos = [247 148 30;0 166 81; 237 28 36; 0 174 239; 0 250 250; 100 50 150; 150 150 200; 0 0 0; 200 30 150; 50 230 50]/255; % colors
for i = 1:length(files)
    name{i} = files(i).name;
end
name = natsortfiles(name);

for k=1:Nfiles
    f = load([Dir, name{k}]);
    t = f(:,1);
    d = f(:,3);
    hold on
    indS = round(5/(k*0.1+0.6)*1000);
    indF = round(8/(k*0.1+0.6)*1000);
    v30(k) = (d(indF) - d(indS)) / (t(indF) - t(indS)) * 1000;
end

Dir = 'C:\Users\huang\Downloads\STAR_frog\Data\sim\freq\20\'

files = dir([Dir, '*.txt']); % Locate the files
Nfiles = length(files);
colpos = [247 148 30;0 166 81; 237 28 36; 0 174 239; 0 250 250; 100 50 150; 150 150 200; 0 0 0; 200 30 150; 50 230 50]/255; % colors
for i = 1:length(files)
    name{i} = files(i).name;
end
name = natsortfiles(name);

for k=1:Nfiles
    f = load([Dir, name{k}]);
    t = f(:,1);
    d = f(:,3);
    hold on
    indS = round(5/(k*0.1+0.3)*1000);
    indF = round(8/(k*0.1+0.3)*1000);
    v20(k) = (d(indF) - d(indS)) / (t(indF) - t(indS)) * 1000;
end

Dir = 'C:\Users\huang\Downloads\STAR_frog\Data\sim\freq\10\'

files = dir([Dir, '*.txt']); % Locate the files
Nfiles = length(files);
colpos = [247 148 30;0 166 81; 237 28 36; 0 174 239; 0 250 250; 100 50 150; 150 150 200; 0 0 0; 200 30 150; 50 230 50]/255; % colors
for i = 1:length(files)
    name{i} = files(i).name;
end
name = natsortfiles(name);

for k=1:Nfiles
    f = load([Dir, name{k}]);
    t = f(:,1);
    d = f(:,3);
    hold on
    indS = round(5/(k*0.1+0.3)*1000);
    indF = round(8/(k*0.1+0.3)*1000);
    v10(k) = (d(indF) - d(indS)) / (t(indF) - t(indS)) * 1000;
end

Dir = 'C:\Users\huang\Downloads\STAR_frog\Data\sim\freq\0\'

files = dir([Dir, '*.txt']); % Locate the files
Nfiles = length(files);
colpos = [247 148 30;0 166 81; 237 28 36; 0 174 239; 0 250 250; 100 50 150; 150 150 200; 0 0 0; 200 30 150; 50 230 50]/255; % colors
for i = 1:length(files)
    name{i} = files(i).name;
end
name = natsortfiles(name);

for k=1:Nfiles
    f = load([Dir, name{k}]);
    t = f(:,1);
    d = f(:,3);
    indS = round(5/(k*0.1+0.3)*1000);
    indF = round(8/(k*0.1+0.3)*1000);
    v00(k) = (d(indF) - d(indS)) / (t(indF) - t(indS)) * 1000;
end
f00 = 0.4:0.1:1.1;
n00 = v00./f00;
f10 = 0.4:0.1:1.2;
n10 = v10./f10;
f20 = 0.4:0.1:1.3;
n20 = v20./f20;
f30 = 0.7:0.1:1.4;
n30 = v30./f30;
f40 = 1.5:0.1:1.9;
n40 = v40./f40;
f50 = 1.7:0.1:2.3;
n50 = v50./f50;