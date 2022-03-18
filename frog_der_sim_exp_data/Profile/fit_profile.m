clc
clear all
close all

% Read the data in dirName and store it
%dirProcessed = [dirName(1:end-1), '_processed/']; % This is where processed images will be saved
%matDir = 'matfiles/'; % This is where mat files will be saved
%matFile = [matDir, 'cur08.mat'];
%fprintf('Results will be saved in %s\n', matFile);
Dir = 'C:\Users\huang\Downloads\STAR_frog\Data\Profile\5\'
files = dir([Dir, '*.mat']); % Locate the images
    Nfiles = length(files);
%ImageProcessedData = cell(Nfiles, 1);
for i = 1:length(files)
    name{i} = files(i).name;
end
name = natsortfiles(name);
for k=1:Nfiles
file = name{k}
load([Dir, file])
K{k} = mean;
T{k} = t;
end


% % deactivation
% % x0 = [80, 4, 1, 10]; %deactuation
% % x0 = [13.98, 1.46]; %deactivation
% % deactivation
% x0 = [100, 4, 0.4, 10]; %deactuation
% % x0 = [3, 1]; %deactivation
% for i = 1 : Nfiles
%     K_temp = K{i};
%     T_temp = T{i};
%     [V, I] = min(K_temp);
%     V0 = K_temp(1);
%     Tf = T_temp(end);
%     n1 = V / K_temp(1);
% %     xdata = T_temp(1 : I)';
% %     ydata = K_temp(1 : I)'; %actuation
%     xdata = T_temp(I : end);
%     ydata = K_temp(I : end)'; %deactuation
%     t = T_temp(I);
% %     fun = @(x,xdata,ydata)x(1)./(1+exp(-x(2)*(xdata-x(3))))+ x(4)%+...
%         %x(5)*(x(1)./(1+exp(-x(2)*(T_temp(I)-x(3))))-V + x(4)) %+ x(6)*(x(1)./(1+exp(-x(2)*(T_temp(end)-x(3))))-V0 + x(4))
% %     fun = @(x,xdata)(n1-1)./(1+exp(-x(1)*(xdata-x(2))))*V0+V0 %actuation
% %     fun = @(x,xdata)(1 - n1)./(1+exp(-x(1)*(xdata-x(2))))*V0+n1*V0 %deactuation
% %     noncon = @(x) x(1)./(1+exp(-x(2)*(T_temp(I)-x(3))))+ x(4)-V 
%       noncon = @(x)Con(x,t,Tf,V,V0)
%     x = fmincon(@(x)fun(x,xdata,ydata),x0,[],[],[],[],[],[],noncon);
% %     options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt')    
% %     x = lsqcurvefit(fun,x0,xdata,ydata, [], [], options);
%     Fun = @(x,xdata)x(1)./(1+exp(-x(2)*(xdata-x(3))))+ x(4)
%     figure(i)
%     hold on
%     times = linspace(xdata(1),xdata(end));
%     plot(xdata,ydata,'ko',times,Fun(x,times),'b-')
%     Par(i, :) = x;
%     ta(i) = t;
%     clear x
% end
% save('5_deact_ta.mat', 'ta')
% %      save('3_deact.mat', 'Par')
%     function obj = fun(x,xdata,ydata)
%         Ysim = x(1)./(1+exp(-x(2)*(xdata-x(3))))+ x(4)  % evaluate logistic function at measurement points
%         obj = (ydata-Ysim)*(ydata-Ysim)'  % calculate sum of squared differences between measured and fitted values
%     end
%     function [c, ceq] = Con(x, t, Tf, V, V0)
%         c = [];
%         ceq = [x(1)./(1+exp(-x(2)*(t-x(3))))-V + x(4);...
%             x(1)./(1+exp(-x(2)*(Tf-x(3))))-V0 + x(4)];
%     end
%     function [c, ceq] = ConOne(x, t, V)
%         c = [];
%         ceq = [x(1)./(1+exp(-x(2)*(t-x(3))))-V + x(4)];
%     end

% actuation
x0 = [-80, 100, 0.05, 100]; %actuation
% x0 = [100, 0.05]; %actuation
 for i = 1 : Nfiles
    K_temp = K{i};
    T_temp = T{i};
    [V, I] = min(K_temp);
    V0 = K_temp(1);
    Tf = T_temp(end);
    n1 = V / K_temp(1);
    xdata = T_temp(1 : I);
    ydata = K_temp(1 : I)'; 
    t = T_temp(I);
 
%     fun = @(x,xdata,ydata)x(1)./(1+exp(-x(2)*(xdata-x(3))))+ x(4)%+...
        %x(5)*(x(1)./(1+exp(-x(2)*(T_temp(I)-x(3))))-V + x(4)) %+ x(6)*(x(1)./(1+exp(-x(2)*(T_temp(end)-x(3))))-V0 + x(4))
%     fun = @(x,xdata)(n1-1)./(1+exp(-x(1)*(xdata-x(2))))*V0+V0 %actuation
%     fun = @(x,xdata)(1 - n1)./(1+exp(-x(1)*(xdata-x(2))))*V0+n1*V0 %deactuation
%     noncon = @(x) x(1)./(1+exp(-x(2)*(T_temp(I)-x(3))))+ x(4)-V 
    noncon = @(x)Con(x,t,V,V0)
    x = fmincon(@(x)fun(x,xdata,ydata),x0,[],[],[],[],[],[],noncon);
%     options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt')    
%     x = lsqcurvefit(fun,x0,xdata,ydata, [], [], options);
    Fun = @(x,xdata)x(1)./(1+exp(-x(2)*(xdata-x(3))))+ x(4)
    figure(i)
    hold on
    times = linspace(xdata(1),xdata(end));
    plot(xdata,ydata,'ko',times,Fun(x,times),'b-')
    par(i, :) = x;
    ta(i) = t;
    clear x
 end
 save('5_act_ta.mat', 'ta')
%     save('1_act.mat', 'par')
    function obj = fun(x,xdata,ydata)
        Ysim = x(1)./(1+exp(-x(2)*(xdata-x(3))))+ x(4)  % evaluate logistic function at measurement points
        obj = (ydata-Ysim)*(ydata-Ysim)'  % calculate sum of squared differences between measured and fitted values
    end
    function [c, ceq] = Con(x, t, V, V0)
        c = [];
        ceq = [x(1)./(1+exp(-x(2)*(t-x(3))))-V + x(4);...
            x(1)./(1+exp(-x(2)*(0-x(3))))-V0 + x(4)];
    end
