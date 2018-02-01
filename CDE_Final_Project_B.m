close all; clearvars; clc;
 

X = readtable('ML_Final_Data_extensive2.xls');



X(:,1) = []; %get rid of observation names, first column
 
%% 

Data = table2array(X);
Variable_Names = X.Properties.VariableNames;

% Remove all data points that have NaN in them
ii = 1;
while ii <= size(Data,1)
    jj = 1;
        while jj <= size(Data,2)
            if ii <= size(Data,1)
                if isnan(Data(ii,jj))
                    Data(ii,:) = [];
                    jj = 1;
                else
                    jj = jj+1;
                end
            else
                break
            end
        end
    ii = ii+1;
end
%%

for ii=1:1:size(Data,2)
    Data(:,ii) = (Data(:,ii)-min(Data(:,ii)))/(max(Data(:,ii))-min(Data(:,ii)));
end

%% 

Data = array2table(Data);
Data.Properties.VariableNames = Variable_Names;
lData = Data;

Y = Data(:,18); %black graduate rate column
Y = table2array(Y);

Data(:,17:19) = []; %delete women, black, and hispanic graduate rate columns

Variable_Names = Data.Properties.VariableNames;
Data = table2array(Data);

%%

D = x2fx(Data,'linear');
D(:,1) = [];
[B,FitInfo] = lasso(D,Y,'CV',10,'PredictorNames',Variable_Names);
minMSEModel = FitInfo.PredictorNames(B(:,FitInfo.IndexMinMSE)~=0)
sparseModel = FitInfo.PredictorNames(B(:,FitInfo.Index1SE)~=0)

%%
lassoPlot(B,FitInfo,'PredictorNames',Variable_Names);
% lassoPlot(B,FitInfo,'PlotType','CV');
% figure; plot(flipud(B'))


%%

mdl = fitlm(lData,'linear','ResponseVar','GraduationRate_Bachelor_sDegreeWithin6Years_Black_Non_Hispanic','PredictorVars',minMSEModel)
