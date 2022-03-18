clear
close all
filenames = {'line1', 'sin1', 'ellipse5', 'line2', 'sin4', 'ellipse4'};
path_filenames = {'linep1', 'sinp1', 'ellipsep5', 'linep2', 'sinp4', 'ellipsep4'};
%filenames = {'cv_datalogger_2022-2-10_154831','cv_datalogger_2022-2-10_155044','cv_datalogger_2022-2-10_155357'};
%path_filenames = {'cv_datalogger_path_2022-2-10_154831','cv_datalogger_path_2022-2-10_155044','cv_datalogger_path_2022-2-10_155357'};
hold on


%% Cost function components

for i = 1:length(filenames)
    T = readtable(strcat(filenames{i},'.csv'));
    
    data = table2array(T(:,1:end-1));
    time0 = data(:,1);
    time{i} = time0 - time0(1);
    cost{i} = data(:,end-3);
    tables{i} = data;
    plot(time{i}/1000,cost{i},'LineWidth',2)
    xlabel('Time (s)')
    ylabel('Cost')
    set(gca,'FontSize',15)
    legend('Line 1', 'Sin 1', 'Ellipse 1', 'Line 2', 'Sin 2', 'Ellipse 2')%, 'LWR Line', 'LWR Sin')
    
end
hold off
%saveas(gcf,'cost.pdf')

C = linspecer(6);
cost_names = {'Distance', 'Angle', 'Progression'};
for j = 1:3
    
    figure
    hold on
    for i = 1:length(filenames)
        T = readtable(strcat(filenames{i},'.csv'));

        data = table2array(T(:,1:end-1));
        time0 = data(:,1);
        time{i} = time0 - time0(1);
        cost{i} = data(:,end-3+j)/500*100;
        tables{i} = data;
        plot(time{i}/1000,cost{i},'LineWidth',2)
        xlabel('Time (s)')
        ylabel(strcat(cost_names{j},' Cost'))
        set(gca,'FontSize',15)
        %set(gcf, 'Position',  [100, 100, 600, 580])
        legend('Line 1', 'Line 2', 'Sin 1', 'Sin 2', 'Ellipse 1', 'Ellipse 2')%, 'LWR Line', 'LWR Sin')
    end
   %saveas(gcf,strcat(cost_names{j},'_cost.pdf'))
end

%% Robot position vs desired path

C = linspecer(6);
for i = 1:3
    figure
    hold on
    P1 = readmatrix(strcat(path_filenames{i},'.csv'));
    P2 = readmatrix(strcat(path_filenames{i+3},'.csv'));
    plot(P1(:,1)*100,P1(:,2)*100, '--','LineWidth',2,'Color',C(i*2-1,:))
    plot(P2(:,1)*100,P2(:,2)*100, '--','LineWidth',2,'Color',C(i*2,:))
    data1 = tables{i};
    data2 = tables{i+3};
    plot(data1(:,2)*100,data1(:,3)*100,'LineWidth',2,'Color',C(i*2-1,:))
    plot(data2(:,2)*100,data2(:,3)*100,'LineWidth',2,'Color',C(i*2,:))
    xlabel('x (cm)')
    ylabel('y (cm)')
    set(gca,'FontSize',15)
    set(gcf, 'Position',  [100, 100, 740, 580])
    axis equal
    %saveas(gcf,strcat('path_plot_',num2str(i),'.pdf'))
    hold off
    
end

%% Distance from path computation
figure
for i=1:3
    hold on
    P1 = readmatrix(strcat(path_filenames{i},'.csv'));
    P2 = readmatrix(strcat(path_filenames{i+3},'.csv'));
    data1 = tables{i};
    time1 = time{i};
    data2 = tables{i+3};
    time2 = time{i+3};
    clear dist1
    clear dist2
    for j = 1:length(data1)
        dist1(j) = min(vecnorm(data1(j,2:3) - P1(:,1:2),2,2));
    end
    for j = 1:length(data2)
        dist2(j) = min(vecnorm(data2(j,2:3) - P2(:,1:2),2,2));
    end
    plot(time1/1000,dist1*100,'LineWidth',2,'Color',C(i*2-1,:))
    plot(time2/1000,dist2*100,'LineWidth',2,'Color',C(i*2,:))
    ylabel('Distance from path (cm)')
    xlabel('Time (s)')
    set(gca,'FontSize',8)
    set(gcf, 'Position',  [100, 100, 600, 580])
    legend('Line 1', 'Line 2', 'Sin 1', 'Sin 2', 'Ellipse 1', 'Ellipse 2')
end
%saveas(gcf,'distances.pdf')

%% Path Overlays
% close all
% for i = 1:3
%     P = readmatrix(strcat(path_filenames{i},'.csv'));
%     figure
%     plot(P(:,1),P(:,2), '--','LineWidth',2,'Color',C(1,:))
%     axis equal
%     set(gcf, 'Position',  [0, 0, 640, 480])
%     xlabel('x')
%     ylabel('y')
%     zlabel('z')
%     set(gca,'xtick',[])
%     set(gca,'xticklabel',[])
%     set(gca,'ytick',[])
%     set(gca,'yticklabel',[])
%     set(gca,'ztick',[])
%     set(gca,'zticklabel',[])
%     set(gca,'XColor','none')
%     set(gca,'YColor','none')
%     set(gca, 'color', 'none');
%     xlabel([])
%     ylabel([])
% 
%     save_path = strcat('path', num2str(i));
%     export_fig(save_path, '-pdf');
% end

%% Mean Velocities:
for i=1:6
    data = tables{i};
    velocity = data(:,5:6);
    norm_v = vecnorm(velocity,2,2);
    mean_v_array(i) = mean(norm_v);
end
mean_v = mean(mean_v_array)
